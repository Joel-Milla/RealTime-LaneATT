import os 
import random
import torch
import utils
import yaml

import numpy as np
import torch.nn as nn

from torchvision import models
from tqdm import tqdm, trange

class LaneATT(nn.Module):
    def __init__(self, config_file=os.path.join(os.path.dirname(__file__), 'config', 'laneatt.yaml')) -> None:
        super(LaneATT, self).__init__()

        # Config files
        self.__laneatt_config_path = config_file
        self.__laneatt_config = yaml.safe_load(open(config_file))

        # Load backbones config file
        self.__backbones_config = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), 'config', 'backbones.yaml')))

        # Set anchor feature channels
        self.__feature_volume_channels = self.__laneatt_config['feature_volume_channels']

        # Set anchor y discretization
        self.__anchor_y_discretization = self.__laneatt_config['anchor_discretization']['y']

        # Set anchor x steps
        self.__anchor_x_discretization = self.__laneatt_config['anchor_discretization']['x']

        # Set image width and height
        self.__img_w = self.__laneatt_config['image_size']['width']
        self.__img_h = self.__laneatt_config['image_size']['height']

        # Create anchor feature dimensions variables but they will be defined after the backbone is created
        self.__feature_volume_height = None
        self.__feature_volume_width = None

        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Creates the backbone and moves it to the device
        self.backbone = self.__laneatt_config['backbone']

        # Generate Anchors Proposals
        self.__anchors_image, self.__anchors_feature_volume = utils.generate_anchors(y_discretization=self.__anchor_y_discretization, 
                                                                                    x_discretization=self.__anchor_x_discretization,
                                                                                    left_angles=self.__laneatt_config['anchor_angles']['left'],
                                                                                    right_angles=self.__laneatt_config['anchor_angles']['right'],
                                                                                    bottom_angles=self.__laneatt_config['anchor_angles']['bottom'],
                                                                                    fv_size=(self.__feature_volume_channels, 
                                                                                             self.__feature_volume_height, 
                                                                                             self.__feature_volume_width),
                                                                                    img_size=(self.__img_h, self.__img_w))
        
        # Move the anchors to the device
        self.__anchors_image = self.__anchors_image.to(self.device)
        self.__anchors_feature_volume = self.__anchors_feature_volume.to(self.device)

        # Pre-Compute Indices for the Anchor Pooling
        self.__anchors_z_indices, self.__anchors_y_indices, self.__anchors_x_indices, self.__invalid_mask = utils.get_fv_anchor_indices(self.__anchors_feature_volume,
                                                                                                                                        self.__feature_volume_channels, 
                                                                                                                                        self.__feature_volume_height, 
                                                                                                                                        self.__feature_volume_width)

        # Move the indices to the device
        self.__anchors_z_indices = self.__anchors_z_indices.to(self.device)
        self.__anchors_y_indices = self.__anchors_y_indices.to(self.device)
        self.__anchors_x_indices = self.__anchors_x_indices.to(self.device)
        self.__invalid_mask = self.__invalid_mask.to(self.device)

        # Fully connected layer of the attention mechanism that takes a single anchor proposal for all the feature maps as input and outputs a score 
        # for each anchor proposal except itself. The score is computed using a softmax function.
        self.__attention_layer = nn.Sequential(nn.Linear(self.__feature_volume_channels * self.__feature_volume_height, len(self.__anchors_feature_volume) - 1),
                                                nn.Softmax(dim=1)).to(self.device)
        
        # Convolutional layer for the classification and regression tasks
        self.__cls_layer = nn.Linear(2 * self.__feature_volume_channels * self.__feature_volume_height, 2).to(self.device)
        self.__reg_layer = nn.Linear(2 * self.__feature_volume_channels * self.__feature_volume_height, self.__anchor_y_discretization + 1).to(self.device)

    @property
    def backbone(self):
        return self.__backbone
    
    @backbone.setter
    def backbone(self, value):
        """
            Set the backbone for the model taking into account available backbones in the config file
            It cuts the average pooling and fully connected layer from the backbone and adds a convolutional 
            layer to reduce the dimensionality to the desired feature volume channels and moves the model 
            to the device
        """
        # Lower the value to avoid case sensitivity
        value = value.lower()

        # Check if value is in the list of backbones in config file
        if value not in self.__backbones_config['backbones']:
            raise ValueError(f'Backbone must be one of {self.config["backbones"]}')
        
        # Set pretrained backbone according to pytorch requirements without the average pooling and fully connected layer
        self.__backbone = nn.Sequential(*list(models.__dict__[value](weights=f'{value.replace("resnet", "ResNet")}_Weights.DEFAULT').children())[:-2],)

        # Runs backbone (on cpu) once to get output data 
        backbone_dimensions = self.__backbone(torch.randn(1, 3, self.__img_h, self.__img_w)).shape

        # Extracts feature volume height and width
        self.__feature_volume_height = backbone_dimensions[2]
        self.__feature_volume_width = backbone_dimensions[3]

        # Join the backbone and the convolutional layer for dimensionality reduction
        self.__backbone = nn.Sequential(self.__backbone, nn.Conv2d(backbone_dimensions[1], self.__feature_volume_channels, kernel_size=1))

        # Move the model to the device
        self.__backbone.to(self.device)

    def forward(self, x):
        """
            Forward pass of the model

            Args:
                x (torch.Tensor): Input image

            Returns:
                torch.Tensor: Regression proposals
        """
        # Move the input to the device
        x = x.to(self.device)
        # Gets the feature volume from the backbone with a dimensionality reduction layer
        feature_volumes = self.backbone(x)
        # Extracts the anchor features from the feature volumes
        batch_anchor_features = self.__cut_anchor_features(feature_volumes)
        # Join proposals from all feature volume channels into a single dimension and stacks all the batches
        batch_anchor_features = batch_anchor_features.view(-1, self.__feature_volume_channels * self.__feature_volume_height)

        # Compute attention scores and reshape them to the original batch size
        attention_scores = self.__attention_layer(batch_anchor_features).reshape(x.shape[0], len(self.__anchors_feature_volume), -1)
        # Generate the attention matrix to be used to store the attention scores
        attention_matrix = torch.eye(attention_scores.shape[1], device=x.device).repeat(x.shape[0], 1, 1)
        # Gets the indices of the non diagonal elements of the attention matrix
        non_diag_indices = torch.nonzero(attention_matrix == 0., as_tuple=False)
        # Makes the entire attention matrix to be zero
        attention_matrix[:] = 0
        # Assigns the attention scores to the attention matrix ignoring the self attention scores as they are not calculated
        # This way we can have a matrix with the attention scores for each anchor proposal
        attention_matrix[non_diag_indices[:, 0], non_diag_indices[:, 1], non_diag_indices[:, 2]] = attention_scores.flatten()

        # Reshape the batch anchor features to the original batch size
        batch_anchor_features = batch_anchor_features.reshape(x.shape[0], len(self.__anchors_feature_volume), -1)
        # Computes the attention features by multiplying the anchor features with the attention weights per batch
        # This will give more context based on the probability of the current anchor to be a lane line compared to other frequently co-occurring anchor proposals
        # And adds them into a single tensor implicitly by using a matrix multiplication
        attention_features = torch.bmm(torch.transpose(batch_anchor_features, 1, 2),
                                       torch.transpose(attention_matrix, 1, 2)).transpose(1, 2)

        # Reshape the attention features batches to one batch size
        attention_features = attention_features.reshape(-1, self.__feature_volume_channels * self.__feature_volume_height)
        # Reshape the batch anchor features batches to one batch size
        batch_anchor_features = batch_anchor_features.reshape(-1, self.__feature_volume_channels * self.__feature_volume_height)

        # Concatenate the attention features with the anchor features
        batch_anchor_features = torch.cat((attention_features, batch_anchor_features), dim=1)

        # Predict the class of the anchor proposals
        cls_logits = self.__cls_layer(batch_anchor_features)
        # Predict the regression of the anchor proposals
        reg = self.__reg_layer(batch_anchor_features)

        # Undo joining the proposals from all images into proposals features batches
        cls_logits = cls_logits.reshape(x.shape[0], -1, cls_logits.shape[1])
        reg = reg.reshape(x.shape[0], -1, reg.shape[1])
        
        # Create the regression proposals
        reg_proposals = torch.zeros((*cls_logits.shape[:2], 5 + self.__anchor_y_discretization), device=self.device)
        # Assign the anchor proposals to the regression proposals
        reg_proposals += self.__anchors_image
        # Assign the classification scores to the regression proposals
        reg_proposals[:, :, :2] = cls_logits
        # Adds the regression offsets to the anchor proposals in the regression proposals
        reg_proposals[:, :, 4:] += reg

        return reg_proposals
    
    def __cut_anchor_features(self, feature_volumes):
        """
            Extracts anchor features from the feature volumes

            Args:
                feature_volumes (torch.Tensor): Feature volumes

            Returns:
                torch.Tensor: Anchor features (n_proposals, n_channels, n_height, 1)
        """

        # Gets the batch size
        batch_size = feature_volumes.shape[0]
        # Gets the number of anchor proposals
        anchor_proposals = len(self.__anchors_feature_volume)
        # Builds a tensor to store the anchor features 
        batch_anchor_features = torch.zeros((batch_size, anchor_proposals, self.__feature_volume_channels, self.__feature_volume_height, 1), 
                                            device=self.device)
        
        # Iterates over each batch
        for batch_idx, feature_volume in enumerate(feature_volumes):
            # Extracts features from the feature volume using the anchor indices, the output will be in a single dimension
            # so we reshape it to a new volume with proposals in the channel dimension, fv_channels in the width dimension
            # and fv_height in the height dimension. So the features extracted from each feature map for each proposal
            # will be in the same channel storing the features of the anchor proposals in each proposed index in the height dimension
            rois = feature_volume[self.__anchors_z_indices, 
                                  self.__anchors_y_indices, 
                                  self.__anchors_x_indices].view(anchor_proposals, self.__feature_volume_channels, self.__feature_volume_height, 1)
            
            # Sets to zero the anchor proposals that are outside the feature map to avoid taking the edge values
            rois[self.__invalid_mask] = 0
            # Assigns the anchor features to the batch anchor features tensor
            batch_anchor_features[batch_idx] = rois

        return batch_anchor_features

    def train_model(self, resume=False):
        """
            Train the model
        """
        # Setup the logger
        logger = utils.setup_logging()
        logger.info('Starting training...')

        model = self.to(self.device)

        # Get the optimizer and the scheduler from the config file
        optimizer = getattr(torch.optim, self.__laneatt_config['optimizer']['name'])(model.parameters(), **self.__laneatt_config['optimizer']['parameters'])
        scheduler = getattr(torch.optim.lr_scheduler, self.__laneatt_config['lr_scheduler']['name'])(optimizer, **self.__laneatt_config['lr_scheduler']['parameters'])

        # State the starting epoch
        starting_epoch = 1
        # Load the last training state if the resume flag is set and modify the starting epoch and model
        if resume:
            last_epoch, model, optimizer, scheduler = utils.load_last_train_state(model, optimizer, scheduler)
            starting_epoch = last_epoch + 1
        
        epochs = self.__laneatt_config['epochs']
        train_loader = self.__get_dataloader('train')

        for epoch in trange(starting_epoch, epochs + 1, initial=starting_epoch - 1, total=epochs):
            logger.debug('Epoch [%d/%d] starting.', epoch, epochs)
            model.train()
            pbar = tqdm(train_loader)
            for i, (images, labels) in enumerate(pbar):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = model(images)
                loss, loss_dict_i = model.loss(outputs, labels)

            #     # Backward and optimize
            #     optimizer.zero_grad()
            #     loss.backward()
            #     optimizer.step()

            #     # Scheduler step (iteration based)
            #     scheduler.step()

            #     # Log
            #     postfix_dict = {key: float(value) for key, value in loss_dict_i.items()}
            #     postfix_dict['lr'] = optimizer.param_groups[0]["lr"]
            #     self.exp.iter_end_callback(epoch, max_epochs, i, len(train_loader), loss.item(), postfix_dict)
            #     postfix_dict['loss'] = loss.item()
            #     pbar.set_postfix(ordered_dict=postfix_dict)
            # self.exp.epoch_end_callback(epoch, max_epochs, model, optimizer, scheduler)

    def __get_dataloader(self, split):
        # Create the dataset object based on TuSimple architecture
        train_dataset = utils.LaneDataset(self.__laneatt_config, split)
        # Create the dataloader object
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=self.__laneatt_config['batch_size'],
                                                   shuffle=True,
                                                   num_workers=20,
                                                   worker_init_fn=self.__worker_init_fn_)
        return train_loader

if __name__ == '__main__':
    model = LaneATT()
    model(torch.randn(2, 3, 1280, 720))
    model.train_model()
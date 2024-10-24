from torchvision import models
from tqdm import tqdm, trange

import os 
import random
import torch
import utils
import yaml

import numpy as np
import torch.nn as nn

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
                loss, loss_dict_i = model.__loss(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Scheduler step (iteration based)
                scheduler.step()

                # Log
                postfix_dict = {key: float(value) for key, value in loss_dict_i.items()}
                postfix_dict['lr'] = optimizer.param_groups[0]["lr"]

                line = 'Epoch [{}/{}] - Iter [{}/{}] - Loss: {:.5f} - '.format(epoch, epochs, i, len(train_loader), loss.item())
                line += ' - '.join(['{}: {:.5f}'.format(component, postfix_dict[component]) for component in postfix_dict])
                logger.debug(line)
                
                postfix_dict['loss'] = loss.item()
                pbar.set_postfix(ordered_dict=postfix_dict)

            logger.debug('Epoch [%d/%d] finished.', epoch, epochs)
            if epoch % self.__laneatt_config['model_checkpoint_interval'] == 0:
                utils.save_train_state(epoch, model, optimizer, scheduler)

    def __loss(self, proposals_list, targets, cls_loss_weight=10):
        focal_loss = utils.FocalLoss(alpha=0.25, gamma=2.)
        smooth_l1_loss = nn.SmoothL1Loss()
        cls_loss = 0
        reg_loss = 0
        valid_imgs = len(targets)
        total_positives = 0
        # Iterate over each batch
        for proposals, target in zip(proposals_list, targets):
            # Filter lane targets that do not exist (confidence == 0)
            target = target[target[:, 1] == 1]

            # Match proposals with targets to get useful indices
            with torch.no_grad():
                positives_mask, invalid_offsets_mask, negatives_mask, target_positives_indices = self.__match_proposals_with_targets(self.__anchors_image, target)

            # The model outputs a classification score and a regression offset for each anchor proposal
            # So we select anchors that are most similar to the ground truth lane lines using the positive mask
            positives = proposals[positives_mask]
            num_positives = len(positives)
            total_positives += num_positives
            # Select anchors that are not similar to the ground truth lane lines using the negative mask
            negatives = proposals[negatives_mask]
            num_negatives = len(negatives)

            # Handle edge case of no positives found by setting targets to 0 and comparing to the classification scores for all proposals that should be 0
            # in a perfect model
            if num_positives == 0:
                cls_target = proposals.new_zeros(len(proposals)).long()
                cls_pred = proposals[:, :2]
                cls_loss += focal_loss(cls_pred, cls_target).sum()
                continue

            # Concatenate positives and negatives
            all_proposals = torch.cat((positives, negatives), dim=0)
            # Create a tensor containing 1 for positives and 0 for negatives
            cls_target = proposals.new_zeros(num_positives + num_negatives).long()
            cls_target[:num_positives] = 1.
            # Get the classification scores
            cls_pred = all_proposals[:, :2]

            # Regression from the positive anchors
            reg_pred = positives[:, 4:]
            with torch.no_grad():
                # Extract the targets that are matched with the positive anchors, could be repeated
                target = target[target_positives_indices]
                # Get the start index of the positive anchors and the target
                positive_starts = (positives[:, 2] / self.__img_h * self.__anchor_y_discretization).round().long()
                target_starts = (target[:, 2]  / self.__img_h * self.__anchor_y_discretization).round().long()
                # Adjust the target length according to the start intersection
                target[:, 4] -= positive_starts - target_starts
                # Create a tensor to store an index for each model output
                all_indices = torch.arange(num_positives, device=self.device, dtype=torch.long)
                # Get the end index of the intersection based on the positive start index and the target length
                ends = (positive_starts + target[:, 4] - 1).round().long()
                # Uses the same trick as matching proposals with targets to get the invalid offsets mask
                invalid_offsets_mask = torch.zeros((num_positives, 1 + self.__anchor_y_discretization + 1), device=self.device, dtype=torch.int)  # length + S + pad
                invalid_offsets_mask[all_indices, 1 + positive_starts] = 1
                invalid_offsets_mask[all_indices, 1 + ends + 1] -= 1
                invalid_offsets_mask = invalid_offsets_mask.cumsum(dim=1) == 0
                invalid_offsets_mask = invalid_offsets_mask[:, :-1]
                invalid_offsets_mask[:, 0] = False
                # Get the regression target
                reg_target = target[:, 4:]
                # Get the regression prediction where no intersection is found and assign it to the target
                # This is done to complete incomplete targets and do not consider them in the regression loss
                # Because even though targets are not full height, predictions are made for the full height
                # And we have to counteract this effect
                reg_target[invalid_offsets_mask] = reg_pred[invalid_offsets_mask]

            # # Loss calc
            reg_loss += smooth_l1_loss(reg_pred, reg_target)
            # Get the classification loss per anchor, adds them to get entire loss and divide by the number of positives 
            # It is not very clear why divide by the number of positives, but it might be to compensate for the number of
            # positives in the batch
            cls_loss += focal_loss(cls_pred, cls_target).sum() / num_positives

        # Batch mean
        cls_loss /= valid_imgs
        reg_loss /= valid_imgs

        loss = cls_loss_weight * cls_loss + reg_loss
        return loss, {'cls_loss': cls_loss, 'reg_loss': reg_loss, 'batch_positives': total_positives}

    def __match_proposals_with_targets(self, proposals, targets, t_pos=15., t_neg=20.):
        """
            Match anchor proposals with targets

            Args:
                proposals (torch.Tensor): Anchor proposals projected to the image space
                targets (torch.Tensor): Ground truth lane lines
                t_pos (float): Positive threshold
                t_neg (float): Negative threshold

            Returns:
                torch.Tensor(num_anchor_proposals): A boolean tensor indicating if the anchor proposal is a positive
                torch.Tensor(num_positives, y_discretization+5): A boolean tensor indicating offsets that do not intersect with the target
                torch.Tensor(num_anchor_proposals): A boolean tensor indicating if the anchor proposal is a negative
                torch.Tensor(num_positives): A tensor with the indices of the target matched with the positive anchor proposal
        """
        
        num_proposals = proposals.shape[0]
        num_targets = targets.shape[0]
        # Pad proposals and target for the valid_offset_mask's trick
        proposals_pad = proposals.new_zeros(proposals.shape[0], proposals.shape[1] + 1)
        proposals_pad[:, :-1] = proposals
        proposals = proposals_pad
        targets_pad = targets.new_zeros(targets.shape[0], targets.shape[1] + 1)
        targets_pad[:, :-1] = targets
        targets = targets_pad

        # Repeat targets and proposals to compare all combinations
        proposals = torch.repeat_interleave(proposals, num_targets, dim=0)
        targets = torch.cat(num_proposals * [targets])

        # Get start index of the proposals and targets
        targets_starts = targets[:, 2] / self.__img_h * self.__anchor_y_discretization
        proposals_starts = proposals[:, 2] / self.__img_h * self.__anchor_y_discretization
        # Get the start index for the intersection
        starts = torch.max(targets_starts, proposals_starts).round().long()
        # Get the end index for the target line
        ends = (targets_starts + targets[:, 4] - 1).round().long()
        # Calculate the length of the intersection
        lengths = ends - starts + 1
        # In the edge case where the intersection is negative, we set the start to the end to achieve the valid_offset_mask's trick
        ends[lengths < 0] = starts[lengths < 0] - 1
        # Since we modify the ends, we need to recalculate the lengths
        lengths[lengths < 0] = 0

        # Generate the valid_offsets_mask that will contain the valid intersection between the proposals and the targets for all combinations
        valid_offsets_mask = targets.new_zeros(targets.shape)
        all_indices = torch.arange(valid_offsets_mask.shape[0], dtype=torch.long, device=targets.device)
        # Put a one on the `start` index and a -1 on the `end` index
        # The -1 is subtracted to account for the case where the length is zero
        valid_offsets_mask[all_indices, 5 + starts] = 1.
        valid_offsets_mask[all_indices, 5 + ends + 1] -= 1.
        # Cumsum to get the valid offsets
        # Valid offsets mask before [0, 0, 0, 1, 0, 0, 0, -1, 0, 0, 0, 0]
        # Valid offsets mask after [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        # And parse it to a boolean mask [False, False, False, True, True, True, True, False, False, False, False, False]
        valid_offsets_mask = valid_offsets_mask.cumsum(dim=1) != 0.
        # Get the invalid offsets mask by inverting the valid offsets mask
        invalid_offsets_mask = ~valid_offsets_mask

        # Compute distances between proposals and targets only inside the intersection
        # Proposals and targets errors
        errors = (targets - proposals)
        # Get only the errors that are inside the intersection
        errors = errors * valid_offsets_mask.float()
        # Get the average distance between the proposals and the targets
        distances = torch.abs(errors).sum(dim=1) / (lengths.float() + 1e-9) # Avoid division by zero

        # For those distances where the length is zero, we set the distance to a very high number since we do not want to consider them
        distances[lengths == 0] = 987654.
        # Reshape the invalid offsets mask to separate the proposals and the targets, so we the invalid mask 
        # of all targets compared to all proposals. And can be indexed by [proposal_idx, target_idx]
        invalid_offsets_mask = invalid_offsets_mask.view(num_proposals, num_targets, -1)
        # Reshape the distances to separate the proposals and the targets, so we can index the distances by [proposal_idx, target_idx]
        distances = distances.view(num_proposals, num_targets)  # d[i,j] = distance from proposal i to target j

        # Get the positives and negatives based on the distances
        # This means for each proposal, we get the target with the minimum distance average error and check if it is below the positive threshold,
        # if it is, then it is a positive, otherwise we check if it is above the negative threshold, if it is, then it is a negative.
        # There is a hysteresis between the positive and negative thresholds to avoid uncertain predictions to pass as either positive or negative
        positives = distances.min(dim=1)[0] < t_pos
        negatives = distances.min(dim=1)[0] > t_neg

        # Verify if there are positives
        if positives.sum() == 0:
            # If there are no positives, we set the target positives indices to an empty tensor
            target_positives_indices = torch.tensor([], device=positives.device, dtype=torch.long)
        else:
            # If there are positives, we get the target positives indices by
            # selecting from distances only the proposals with at least one positive (that should be in the positives tensor which is a boolean mask)
            # and get the index of the minimum distance for each proposal.
            target_positives_indices = distances[positives].argmin(dim=1)

        # Finally we update invalid_offsets_mask to only consider the masks of targets that have been matched with a positive proposal
        invalid_offsets_mask = invalid_offsets_mask[positives, target_positives_indices]

        return positives, invalid_offsets_mask[:, :-1], negatives, target_positives_indices

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
    
    @staticmethod
    def __worker_init_fn_(_):
        torch_seed = torch.initial_seed()
        np_seed = torch_seed // 2**32 - 1
        random.seed(torch_seed)
        np.random.seed(np_seed)

if __name__ == '__main__':
    model = LaneATT()
    model.train_model(resume=True)
from . import utils
from torchvision import models
from tqdm import tqdm, trange

import cv2
import os 
import random
import torch
import yaml

import numpy as np
import torch.nn as nn

class LaneATT(nn.Module):
    def __init__(self, config:str) -> None:
        """
            LaneATT model initialization.

            Args:
                config (str): Path to the configuration file
        """
        super(LaneATT, self).__init__()

        # Config file
        self.__laneatt_config = yaml.safe_load(open(config))

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
    def backbone(self) -> nn.Module:
        """
            Get the backbone (RESNET) of the model
        """
        return self.__backbone
    
    @backbone.setter
    def backbone(self, value:str) -> None:
        """
            Set the backbone for the model taking into account available backbones in the config file
            It cuts the average pooling and fully connected layer from the backbone and adds a convolutional 
            layer to reduce the dimensionality to the desired feature volume channels and moves the model 
            to the device

            Args:
                value (str): Backbone name
            
            Raises:
                ValueError: If the backbone is not in the list of backbones in the config file
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

    @property
    def img_w(self) -> int:
        return self.__img_w
    
    @property
    def img_h(self) -> int:
        return self.__img_h

    def forward(self, x:torch.Tensor) -> torch.Tensor:
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
    
    def nms(self, output:torch.Tensor, threshold:float=0.5, nms_threshold:float=40.0) -> torch.Tensor:
        """
            Apply non-maximum suppression to the proposals

            Args:
                output (torch.Tensor): Regression proposals
                threshold (float): Threshold for valid proposals
                nms_threshold (float): NMS threshold

            Returns:
                torch.Tensor: Good proposals NMS suppressed 
        """
        # Filter proposals with confidence below the threshold and sort them by confidence
        good_proposals = output[output[:, 1] > threshold]
        good_proposals = good_proposals[good_proposals[:, 3].argsort(descending=True)]
        # Verify if there are no proposals
        if len(good_proposals) == 0: return good_proposals

        # Create a mask to store the same line proposals
        good_proposals_mask = np.zeros((len(good_proposals), len(good_proposals)), dtype=bool)
        # Iterate over the proposals
        for i, line_a in enumerate(good_proposals):
            # Get the start and end of the current proposal
            start_a = line_a[2] / self.__img_h * self.__anchor_y_discretization
            end_a = line_a[2] + line_a[4]
            # Iterate over the rest of the proposals
            for j, line_b in enumerate(good_proposals):
                # Get the start and end of the current comparison proposal
                start_b = line_b[2] / self.__img_h * self.__anchor_y_discretization
                end_b = line_b[2] + line_b[4] - 1
                
                # Get the start and end intersection between the proposals
                start = int(max(start_a, start_b))
                end = int(min(end_a, end_b, self.__anchor_y_discretization))

                # Calculate the error between the proposals
                error = 0
                for k in range(start, end):
                    error += abs(line_a[5 + k] - line_b[5 + k])
                error /= end - start
                error = error.item()

                # If the error is below the NMS threshold, we consider the proposals to represent the same line
                good_proposals_mask[i][j] = error < nms_threshold

        # List to store the indexes of the unique lines
        unique_line_indexes = [0]
        while True:
            # Get a unique line
            line = good_proposals_mask[unique_line_indexes[-1]]
            found_different = False
            # Iterate over a unique line against the rest of the proposals errors
            for i, cmp_line in enumerate(line):
                # If the line is different and the index is greater than the last unique line index we found a different line
                # so we append it to the unique line indexes
                if not cmp_line and i > unique_line_indexes[-1]:
                    unique_line_indexes.append(i)
                    found_different = True
                    break
            
            # If we stop finding different lines, we break the loop
            if not found_different:
                break

        # Based on the unique line indexes, we get a range of similar lines and get the one with the highest confidence
        # Create a list to store the high confidence unique line indexes
        high_confidence_unique_line_indexes = [0 for _ in range(len(unique_line_indexes))]
        # Iterate over the unique line indexes
        for i in range(len(unique_line_indexes)):
            # Verify if we are in the last unique line index
            if i == len(unique_line_indexes) - 1:
                # If so, we get the highest confidence line from the last unique line index to the end
                high_confidence_unique_line_indexes[i] = good_proposals[unique_line_indexes[i]:][:, 1].argmax().item()
            else:
                # Otherwise, we get the highest confidence line from the current unique line index to the next unique line index
                high_confidence_unique_line_indexes[i] = good_proposals[unique_line_indexes[i]:unique_line_indexes[i+1]][:, 1].argmax().item()
            
            # Add an offset to counteract for the list slicing
            high_confidence_unique_line_indexes[i] += unique_line_indexes[i]
                
        return good_proposals[unique_line_indexes]
                
    def plot(self, output:torch.Tensor, image:np.ndarray) -> None:
        """
            Plot the lane lines on the image

            Args:
                output (torch.Tensor): Regression proposals
                image (np.ndarray): Image
        """
        proposals_length = output[:, 4]
        # Get the y discretization values
        ys = torch.linspace(self.__img_h, 0, self.__anchor_y_discretization, device=self.device)
        # Store x and y values for each lane line
        output = [[(x, ys[i]) for i, x in enumerate(lane[5:])] for lane in output]

        # Resize the image to the model's trained size
        img = cv2.resize(image, (self.__img_w, self.__img_h))
        # Iterate over the lanes
        for i, lane in enumerate(output):
            # Internal loop variables to account for the first point and the change in color of the lines
            prev_x, prev_y = lane[0]
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            # Iterate over the line points
            for j, (x, y) in enumerate(lane):
                # Break the loop if the proposal length is reached
                if int(proposals_length[i].item()) == j: break
                # Draw a line between the previous point and the current point
                cv2.line(img, (int(prev_x), int(prev_y)), (int(x), int(y)), color, 2)
                prev_x, prev_y = x, y

        # Show the image
        cv2.imshow('frame', img)

    def train_model(self, resume:bool=False):
        """
            Train the model
        """
        # Setup the logger
        logger = utils.setup_logging(self.__laneatt_config['logs_dir'])
        logger.info('Starting training...')

        model = self.to(self.device)

        # Get the optimizer and the scheduler from the config file
        optimizer = getattr(torch.optim, self.__laneatt_config['optimizer']['name'])(model.parameters(), **self.__laneatt_config['optimizer']['parameters'])
        scheduler = getattr(torch.optim.lr_scheduler, self.__laneatt_config['lr_scheduler']['name'])(optimizer, **self.__laneatt_config['lr_scheduler']['parameters'])

        # State the starting epoch
        starting_epoch = 1
        # Load the last training state if the resume flag is set and modify the starting epoch and model
        if resume:
            last_epoch, model, optimizer, scheduler = utils.load_last_train_state(model, optimizer, scheduler, self.__laneatt_config['checkpoints_dir'])
            starting_epoch = last_epoch + 1
        
        # Get the number of epochs from the config file and the train loader
        epochs = self.__laneatt_config['epochs']
        train_loader = self.__get_dataloader('train')

        # Iterate over the epochs
        for epoch in trange(starting_epoch, epochs + 1, initial=starting_epoch - 1, total=epochs):
            logger.debug('Epoch [%d/%d] starting.', epoch, epochs)
            # Put the model in training mode
            model.train()
            # Get the progress bar object based on the train loader
            pbar = tqdm(train_loader)
            # Iterate over the batches
            for i, (images, labels) in enumerate(pbar):
                # Move the images and labels to the device
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
            # Save the model state every model_checkpoint_interval epochs from the config file
            if epoch % self.__laneatt_config['model_checkpoint_interval'] == 0:
                utils.save_train_state(epoch, model, optimizer, scheduler, self.__laneatt_config['checkpoints_dir'])

    def load(self, checkpoint:str):
        """
            Load the model from a checkpoint file

            Args:
                checkpoint (str): Checkpoint file
        """
        train_state = torch.load(checkpoint, weights_only=True)
        self.load_state_dict(train_state['model'])

    def __cut_anchor_features(self, feature_volumes:torch.Tensor) -> torch.Tensor:
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

    def __loss(self, proposals_list:torch.Tensor, targets:torch.Tensor, cls_loss_weight:int=10) -> torch.Tensor:
        """
            Compute the loss of the model

            Args:
                proposals_list (torch.Tensor): Tensor of proposals
                targets (torch.Tensor): Tensor of targets

            Returns:
                torch.Tensor: Loss value in Tensor format
        """
        # Initialize the variables for the losses
        focal_loss = utils.FocalLoss(alpha=0.25, gamma=2.)
        smooth_l1_loss = nn.SmoothL1Loss()
        cls_loss = 0
        reg_loss = 0
        valid_imgs = len(targets)
        total_positives = 0
        # Iterate over each batch
        for i, (proposals, target) in enumerate(zip(proposals_list, targets)):
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

        # Total loss
        loss = cls_loss_weight * cls_loss + reg_loss
        return loss, {'cls_loss': cls_loss, 'reg_loss': reg_loss, 'batch_positives': total_positives}

    def __match_proposals_with_targets(self, proposals:torch.Tensor, targets:torch.Tensor, t_pos:int=15., t_neg:int=20.) -> torch.Tensor:
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

    def __get_dataloader(self, split:str) -> torch.utils.data.DataLoader:
        """
            Get the dataloader object

            Args:
                split (str): Split

            Returns:
                torch.utils.data.DataLoader: Dataloader object
        """
        # Create the dataset object based on TuSimple architecture
        train_dataset = utils.LaneDataset(self.__laneatt_config, split)
        # Create the dataloader object
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=self.__laneatt_config['batch_size'],
                                                   shuffle=True,
                                                   worker_init_fn=self.__worker_init_fn_)
        return train_loader
    
    @staticmethod
    def __worker_init_fn_(_):
        """
            Worker initialization function

            Args:
                _ (int): Worker id

            Notes:
                This function is used to set the seed of the workers to avoid randomness
        """
        torch_seed = torch.initial_seed()
        np_seed = torch_seed // 2**32 - 1
        random.seed(torch_seed)
        np.random.seed(np_seed)
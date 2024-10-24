from typing import List

import torch

import numpy as np

def generate_anchors(y_discretization:int, x_discretization:int, left_angles:list, right_angles:list, bottom_angles:list, fv_size:tuple, img_size:tuple) -> List[torch.Tensor]:
    """
        Generates anchors for the model based on the discretization and angles
        The anchors are in the form of a tensor with shape 
        (y_discretization*[len(left_angles)+len(right_angles)]+x_discretization*len(bottom_angles), (y_discretization+5 or feature volume height+5)) 
        where each row contains each anchor information and y_(y_discretization+5 or feature volume height+5) is the number of features for each anchor
        
        Structure of each anchor:
        [score0, score1, start_y, start_x, length, anchor_xs]
        where:
        - score0: No line score
        - score1: Line score
        - start_y: Start y coordinate (bottom is 0, top is image height or feature volume height)
        - start_x: Start x coordinate (left is 0, right is image width or feature volume width)
        - length: Length of the anchor (steps in the discretization)
        - anchor_xs: X coordinates of the anchor points (image pixel coordinates or feature volume pixel coordinates) [Bottom to top]

        Args:
            y_discretization (int): Number of steps in y direction
            x_discretization (int): Number of steps in x direction
            left_angles (list): List of left angles
            right_angles (list): List of right angles
            bottom_angles (list): List of bottom angles
            fv_size (tuple): Feature volume size
            img_size (tuple): Image size
        
        Returns:
            torch.Tensor: Anchors for all sides projected into the image
            torch.Tensor: Anchors for all sides projected into the feature volume
    """

    # Generate left anchors
    left_anchors_image, left_anchors_volume = generate_side_anchors(left_angles, y_discretization, fv_size, y_discretization, img_size, x=0.)
    # Generate right anchors
    right_anchors_image, right_anchors_volume = generate_side_anchors(right_angles, y_discretization, fv_size, y_discretization, img_size, x=1.)
    # Generate bottom anchors
    bottom_anchors_image, bottom_anchors_volume = generate_side_anchors(bottom_angles, x_discretization, fv_size, y_discretization, img_size, y=1.)

    # Concatenate anchors and cut anchors
    return torch.cat([left_anchors_image, bottom_anchors_image, right_anchors_image]), torch.cat([left_anchors_volume, bottom_anchors_volume, right_anchors_volume]) 

def generate_side_anchors(angles:list, discretization:int, fv_size:tuple, y_discretization:int, img_size:tuple, x:float=None, y:float=None) -> List[torch.Tensor]:
    """
        Generates side anchors based on predefined angles, and discretization
        The anchors are in the form of a tensor with shape (discretization*len(angles), y_(y_discretization+5 or feature volume height+5)) 
        where each row contains each anchor information and y_(y_discretization+5 or feature volume height+5) is the number of features for each anchor
        
        Structure of each anchor:
        [score0, score1, start_y, start_x, length, anchor_xs]
        where:
        - score0: No line score
        - score1: Line score
        - start_y: Start y coordinate (bottom is 0, top is image height or feature volume height)
        - start_x: Start x coordinate (left is 0, right is image width or feature volume width)
        - length: Length of the anchor (steps in the discretization)
        - anchor_xs: X coordinates of the anchor points (image pixel coordinates or feature volume pixel coordinates) [Bottom to top]

        Args:
            angles (list): List of angles
            discretization (int): Number of steps in the side direction
            fv_size (tuple): Feature volume size
            y_discretization (int): Number of steps in y direction
            img_size (tuple): Image size
            x (float): X coordinate of the side
            y (float): Y coordinate of the side

        Returns:
            torch.Tensor: Anchors for the side projected into the image
            torch.Tensor: Anchors for the side projected into the feature volume
    """

    # Check if x or y is None
    if x is None and y is not None:
        # Generate starts based on a fixed y
        starts = [(x, y) for x in np.linspace(1., 0., num=discretization)]
    elif x is not None and y is None:
        # Generate starts based on a fixed x
        starts = [(x, y) for y in np.linspace(1., 0., num=discretization)]
    else:
        # Raises an error if no side is defined
        raise Exception('Please define exactly one of `x` or `y` (not neither nor both)')

    # Calculate number of anchors since one anchor is generated for each angle and origin
    anchors_number = discretization * len(angles)

    # Initialize anchors as a tensor of anchors_number as rows and (y_discretization or feature map height + 5) as columns.
    # This represents each anchor list will have 2 scores, 1 start y, 1 start x, 1 length and y_discretization or feature map height x coordinates
    anchors_image = torch.zeros((anchors_number, 2 + 2 + 1 + y_discretization))
    anchors_feature_volume = torch.zeros((anchors_number, 2 + 2 + 1 + fv_size[1]))

    # Iterates over each start point
    for i, start in enumerate(starts):
        # Iterates over each angle for each start point
        for j, angle in enumerate(angles):
            # Calculates the index of the anchor
            k = i * len(angles) + j
            # Generates the anchors
            anchors_image[k] = generate_anchor(start, angle, y_discretization, fv_size, img_size)
            anchors_feature_volume[k] = generate_anchor(start, angle, y_discretization, fv_size, img_size, fv=True)

    return anchors_image, anchors_feature_volume

def generate_anchor(start:tuple, angle:float, y_discretization:int, fv_size:int, img_size:tuple, fv:bool=False) -> torch.Tensor:
    """
        Generates anchor based on start point and angle
        The anchors are in the form of a tensor with shape (1, y_(y_discretization+5 or feature volume height+5)) 
        where y_(y_discretization+5 or feature volume height+5) is the number of features for each anchor
        
        Structure of each anchor:
        [score0, score1, start_y, start_x, length, anchor_xs]
        where:
        - score0: No line score
        - score1: Line score
        - start_y: Start y coordinate (bottom is 0, top is image height or feature volume height)
        - start_x: Start x coordinate (left is 0, right is image width of feature volume width)
        - length: Length of the anchor (steps in the discretization)
        - anchor_xs: X coordinates of the anchor points (image pixel coordinates) [Bottom to top]

        Args:
            start (tuple): Start point (x -> left is 0, right is 1, y -> top is 1, bottom is 0)
            angle (float): Angle of the anchor
            y_discretization (int): Number of steps in y direction
            fv_size (tuple): Feature volume size
            img_size (tuple): Image size
            fv (bool): Flag to indicate if the anchor is projected into the feature volume

        Returns:
            torch.Tensor: Anchor projected into the image or feature volume
    """

    # Extract image width and height
    img_h, img_w = img_size
    # Extract feature volume data
    _, fv_h, fv_w = fv_size
    # Convert angle to radians
    angle = angle * np.pi / 180.
    # Extract start x and y from start point
    start_x, start_y = start

    # Check if fv is True
    if fv:
        # Set anchor y coordinates from 1 to 0 with feature map height steps
        anchor_ys = torch.linspace(1, 0, steps=fv_h, dtype=torch.float32)
        # Initialize anchor tensor with 2 scores, 1 start y, 1 start x, 1 length and feature volume height
        anchor = torch.zeros(2 + 2 + 1 + fv_h)
        # Assigns to third element of anchor tensor the start y in feature volume coordinates
        anchor[2] = (1 - start_y)  * fv_h
        # Assigns to fourth element of anchor tensor the start x in feature volume coordinates
        anchor[3] = start_x * fv_w
        # Gets a relative delta y based on the start coordinate for each fv height point
        delta_y = (start_y - anchor_ys) * fv_h
    else:
        # Set anchor y coordinates from 1 to 0 with n_offsets steps
        anchor_ys = torch.linspace(1, 0, steps=y_discretization, dtype=torch.float32)
        # Initialize anchor tensor with 2 scores, 1 start y, 1 start x, 1 length and discretization points
        anchor = torch.zeros(2 + 2 + 1 + y_discretization)
        # Assigns to third element of anchor tensor the start y in image coordinates
        anchor[2] = (1 - start_y)  * img_h
        # Assigns to fourth element of anchor tensor the start x in image coordinates
        anchor[3] = start_x * img_w
        # Gets a relative delta y based on the start coordinate for each discretization points
        delta_y = (start_y - anchor_ys) * img_h

    # Gets a relative delta x from the origin point for each anchor point based on the angle and delta y since -> 1/tan(angle) = delta x / delta y
    delta_x = delta_y / np.tan(angle)
    # Adds the delta x of each anchor point to the start x to get the x coordinate of each anchor point
    anchor[5:] = (anchor[3] + delta_x)

    return anchor

def get_fv_anchor_indices(anchors_fv, feature_map_channels, feature_volume_height, feature_volume_width):
        """
            Computes the anchor indices for the feature volume

            Args:
                anchors_fv (torch.Tensor): Anchors for all sides projected into the feature volume
                feature_map_channels (int): Number of feature map channels
                feature_volume_height (int): Feature volume height
                feature_volume_width (int): Feature volume width
            
            Returns:
                torch.Tensor: Z indices ((0-64)*fv_height[interleaved]*n_proposals, 1)
                torch.Tensor: Y indices ((0-23)*fv_channels*n_proposals, 1)
                torch.Tensor: X indices (ith_proposal[5:]*fv_channels*n_proposals, 1)
                torch.Tensor: Invalid mask (ith_proposal[5:]*fv_channels*n_proposals, 1)
        """
        # Get the number of anchors proposals
        n_proposals = len(anchors_fv)

        # Extract only anchor points from anchors_feature_volume tensor
        anchors_x_points = anchors_fv[:, 5:]
        # Repeat the anchors proposals for each feature map and puts them in a single dimension
        # The output of this line stores the ith anchor proposal repeated for each feature map channel collapsed in a single dimension
        # (ith_propsal[5:], fv_h) -> (ith_proposal[5:]*fv_channels, 1) and rounded to the nearest integer to get the nearest pixel index
        unclamped_anchors_x_indices = torch.repeat_interleave(anchors_x_points, feature_map_channels, dim=0).reshape(-1, 1).round().long()
        # Clamp the anchors coordinates to the feature map width
        anchors_x_indices = torch.clamp(unclamped_anchors_x_indices, 0, feature_volume_width - 1)
        # Reshape the anchors to the original shape
        unclamped_anchors_x_indices = unclamped_anchors_x_indices.reshape(n_proposals, feature_map_channels, feature_volume_height, 1)
        
        # Generate a binary mask to filter out the invalid anchor proposals
        invalid_mask = (unclamped_anchors_x_indices < 0) | (unclamped_anchors_x_indices > feature_volume_width - 1)

        # Generate y coordinates for each anchor point
        anchors_y_indices = torch.arange(0, feature_volume_height)
        # Repeat the y coordinates for each feature map and each proposal and puts them in a single dimension
        # ((0-23)*fv_channels*n_proposals, 1)
        anchors_y_indices = anchors_y_indices.repeat(feature_map_channels).repeat(n_proposals).reshape(-1, 1)

        # Generate z pixels for each anchor proposal and puts them in a single dimension
        # ((0-64)*fv_height[interleaved]*n_proposals, 1)
        anchors_z_cut_indices = torch.arange(feature_map_channels).repeat_interleave(feature_volume_height).repeat(n_proposals).reshape(-1, 1)

        return anchors_z_cut_indices, anchors_y_indices, anchors_x_indices, invalid_mask
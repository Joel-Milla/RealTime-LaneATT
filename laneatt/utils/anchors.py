import torch
import numpy as np

def generate_anchors(y_discretization, x_discretization, left_angles, right_angles, bottom_angles, feature_volume_height, img_size):
    """
        Generates anchors for the model based on the discretization and angles
        The anchors are in the form of a tensor with shape 
        (y_discretization*[len(left_angles)+len(right_angles)]+x_discretization*len(bottom_angles), 77) 
        where each row contains each anchor information and 77 is the number of features for each anchor
        
        Structure of each anchor:
        [score0, score1, starty, startx, length, anchor_xs]
        where:
        - score0: No line score
        - score1: Line score
        - starty: Start y coordinate (bottom is 0, top is image height)
        - startx: Start x coordinate (left is 0, right is image width)
        - length: Length of the anchor (steps in the discretization)
        - anchor_xs: X coordinates of the anchor points (image pixel coordinates)
        - img_size: Image size (height, width)

        Args:
            y_discretization (int): Number of lateral anchors
            x_discretization (int): Number of bottom anchors
            left_angles (list): List of left angles
            right_angles (list): List of right angles
            bottom_angles (list): List of bottom angles
            feature_volume_height (int): Height of the feature map

        Returns:
            torch.Tensor: Anchors for all sides projected into the image
            torch.Tensor: Anchors for all sides projected into the feature volume
    """

    # Generate left anchors
    left_anchors_image, left_anchors_volume = generate_side_anchors(left_angles, y_discretization, feature_volume_height, y_discretization, img_size, x=0.)
    # Generate right anchors
    right_anchors_image, right_anchors_volume = generate_side_anchors(right_angles, y_discretization, feature_volume_height, y_discretization, img_size, x=1.)
    # Generate bottom anchors
    bottom_anchors_image, bottom_anchors_volume = generate_side_anchors(bottom_angles, x_discretization, feature_volume_height, y_discretization, img_size, y=1.)

    # Concatenate anchors and cut anchors
    return torch.cat([left_anchors_image, bottom_anchors_image, right_anchors_image]), torch.cat([left_anchors_volume, bottom_anchors_volume, right_anchors_volume]) 

def generate_side_anchors(angles, discretization, feature_volume_height, y_discretization, img_size, x=None, y=None):
    """
        Generates side anchors based on predefined angles, and discretization

        Args:
            angles (list): List of angles
            discretization (int): Number of origins
            feature_volume_height (int): Height of the feature map
            x (float): X coordinate to define the side
            y (float): Y coordinate to define the side
            y_discretization (int): Number of steps in y direction

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
    anchors_feature_volume = torch.zeros((anchors_number, 2 + 2 + 1 + feature_volume_height))

    # Iterates over each start point
    for i, start in enumerate(starts):
        # Iterates over each angle for each start point
        for j, angle in enumerate(angles):
            # Calculates the index of the anchor
            k = i * len(angles) + j
            # Generates the anchors
            anchors_image[k] = generate_anchor(start, angle, y_discretization, feature_volume_height, img_size)
            anchors_feature_volume[k] = generate_anchor(start, angle, y_discretization, feature_volume_height, img_size, fv=True)

    return anchors_image, anchors_feature_volume

def generate_anchor(start, angle, y_discretization, feature_volume_height, img_size, fv=False):
    """
        Generates anchor based on start point and angle

        Args:
            start (tuple): Start point
            angle (float): Angle
            y_discretization (int): Number of steps in y direction
            feature_volume_height (int): Height of the feature map
            fv (bool): If True, the anchor will be projected into the feature volume

        Returns:
            torch.Tensor: Anchor
    """

    # Check if fv is True
    if fv:
        # Set anchor y coordinates from 1 to 0 with feature map height steps
        anchor_ys = torch.linspace(1, 0, steps=feature_volume_height, dtype=torch.float32)
        # Initialize anchor tensor with 2 scores, 1 start y, 1 start x, 1 length and feature map height
        anchor = torch.zeros(2 + 2 + 1 + feature_volume_height)
    else:
        # Set anchor y coordinates from 1 to 0 with n_offsets steps
        anchor_ys = torch.linspace(1, 0, steps=y_discretization, dtype=torch.float32)
        # Initialize anchor tensor with 2 scores, 1 start y, 1 start x, 1 length and n_offsets
        anchor = torch.zeros(2 + 2 + 1 + y_discretization)
    # Extract image width and height
    img_h, img_w = img_size
    # Convert angle to radians
    angle = angle * np.pi / 180.
    # Extract start x and y from start point
    start_x, start_y = start
    # Assigns to third element of anchor tensor the start y taking the bottom as 0
    anchor[2] = (1 - start_y) * img_h
    # Assigns to fourth element of anchor tensor the start x
    anchor[3] = start_x * img_w
    # Gets a relative delta y based on the start coordinate for each n_offsets points
    delta_y = (anchor_ys - start_y)
    # Gets a relative delta x from the origin point for each anchor point based on the angle and delta y since -> 1/tan(angle) = delta x / delta y
    delta_x = delta_y / np.tan(angle)
    # Adds the delta x of each anchor point to the start x to get the x coordinate of each anchor point
    anchor[5:] = (start_x + delta_x) * img_w

    return anchor
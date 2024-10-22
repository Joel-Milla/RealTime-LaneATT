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

if __name__ == '__main__':
    model = LaneATT()
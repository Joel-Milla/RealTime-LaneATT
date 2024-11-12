from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from scipy.interpolate import InterpolatedUnivariateSpline

import cv2
import json
import logging
import os
import random

import numpy as np

class LaneDataset(Dataset):
    def __init__(self, config:str, split:str) -> None:
        """
            Initialize the LaneDataset object.

            Args:
                config: The path to the configuration file.
                split: The dataset split to load, e.g. train, val, test.
        """
        self.__general_config = config
        self.__dataset_config = config['dataset'][split]
        # Dataset split to load, e.g. train, val, test
        self.__split = split
        # The root directory of the dataset
        self.__root = self.__dataset_config['root']
        # A logger object
        self.__logger = logging.getLogger(__name__)

        # Image width and height (model's input size and dataset's image size)
        self.img_w, self.img_h = self.__general_config['image_size']['width'], self.__general_config['image_size']['height']
        self.dataset_img_w, self.dataset_img_h = self.__general_config['dataset_image_size']['width'], self.__general_config['dataset_image_size']['height']

        # Load the annotation files
        jsons = os.listdir(self.__root)
        jsons = [j for j in jsons if j.endswith('.json')]
        self.__annotation_files = [os.path.join(self.__root, json) for json in jsons]

        # Verify that the root directory is specified
        if self.__root is None:
            raise Exception('Please specify the root directory')

        # Load the annotations
        self.annotations = []
        self.max_lanes = 0
        self.__load_annotations()

        # Transform the annotations to the model's target format
        self.__logger.info("Transforming annotations to the model's target format...")
        self.__y_discretizations = self.__general_config['anchor_discretization']['y']
        self.__ys = np.linspace(1, 0, self.__y_discretizations)
        self.annotations = np.array(list(map(self.__transform_annotation, self.annotations)))
        self.__logger.info("Annotations transformed.")

    def __load_annotations(self) -> None:
        """
            Load annotations from the annotation files as a dict of the path of the image and a list of lists of tuples (x,y) for the lanes.

            annotations: {
                            path: str,
                            lanes: list[list[tuple(float, float)]]
                            }
        """

        self.__logger.info('Loading TuSimple annotations...')
        max_lanes = 0
        self.annotations = []
        # Iterate over the annotation files
        for annotation_file in self.__annotation_files:
            # Opens and reads the annotation file
            with open(annotation_file, 'r') as annotation_obj:
                lines = annotation_obj.readlines()
            # Iterate over the lines in the annotation file
            for line in lines:
                # Load the JSON line data
                try:
                    data = json.loads(line)
                except:
                    print('Error loading JSON line:', line)
                # Get the lanes y coordinates
                y_samples = data['h_samples']
                # Get the lanes x coordinates
                og_lanes = data['lanes']
                # Create the lanes as a tuple of x and y coordinates
                lanes = [[(x, y) for (x, y) in zip(lane, y_samples) if x >= 0] for lane in og_lanes]
                # Get the maximum number of lanes in an image
                max_lanes = max(max_lanes, len(lanes))

                # Append the annotation to the list of annotations
                self.annotations.append({
                    'path': os.path.join(self.__root, data['raw_file']),
                    'lanes': lanes,
                })

        # Shuffle the annotations if the split is train
        if self.__split == 'train':
            random.shuffle(self.annotations)
        self.max_lanes = max_lanes
        self.__logger.info('%d annotations loaded, with a maximum of %d lanes in an image.', len(self.annotations),
                         self.max_lanes)

    def __transform_annotation(self, annotation:dict) -> dict:
        """
            Transform the annotation to the model's target format.
            
            The transformed lanes involve cleaning the lanes, sorting them by Y, removing points with the same Y, 
            remapping the lanes to a 0-1 range, and creating a new annotation with the transformed lanes and the model
            y_discretizations based on polynomial interpolations and extrapolations of the labeled lanes.

            It also involves labeling valid lanes in the score fields, setting the starting point of the lanes (based on
            the first x inside the image and its corresponding y assuming bottom approximations only can leave the image
            by one side), and its length.

            Args:
                annotation: A dictionary containing the path of the image and the lanes.

            Returns:
                A dictionary containing the path of the image and the transformed lanes.Returns:
                A dictionary containing the path of the image and the transformed lanes.
        """
        old_lanes = annotation['lanes']

        # Remove lanes with less than 2 points
        old_lanes = filter(lambda x: len(x) > 1, old_lanes)
        # Sort lane points by Y (bottom to top of the image)
        old_lanes = [sorted(lane, key=lambda x: -x[1]) for lane in old_lanes]
        # Remove points with same Y (keep first occurrence)
        old_lanes = [self.__filter_lane(lane) for lane in old_lanes]
        # Remap the lanes to the model config image size
        old_lanes = [[(x/self.dataset_img_w, y/self.dataset_img_h) for (x, y) in lane] for lane in old_lanes]
        # Create tranformed annotations placeholder
        lanes = np.ones((self.max_lanes, 2 + 1 + 1 + 1 + self.__y_discretizations),
                        dtype=np.float32) * -1e5
        
        # Lanes are invalid by default
        lanes[:, 0] = 1
        lanes[:, 1] = 0
        # Iterates over the lanes
        for lane_idx, lane in enumerate(old_lanes):
            try:
                xs_outside_image, xs_inside_image = self.__sample_lane(lane)
            except AssertionError:
                continue
            if len(xs_inside_image) == 0:
                continue
            # Joins the points outside and inside the image again
            all_xs = np.hstack((xs_outside_image, xs_inside_image))

            # Updates the lanes with the new points
            # Sets the scores to 0 for no lane and 1 for lane
            lanes[lane_idx, 0] = 0
            lanes[lane_idx, 1] = 1
            # Set y starting point bottom to top
            lanes[lane_idx, 2] = (1-self.__ys[len(xs_outside_image)]) * self.img_h
            # Set x starting point
            lanes[lane_idx, 3] = xs_inside_image[0] * self.img_w
            # Set the number of points in the lane
            lanes[lane_idx, 4] = len(xs_inside_image)
            # Set the x coordinates of the lane
            lanes[lane_idx, 5:5 + len(all_xs)] = all_xs * self.img_w

        new_annotation = {'path': annotation['path'], 'label': lanes}
        return new_annotation
    
    def __filter_lane(self, lane:list) -> list:
        """
            Remove points with the same Y coordinate, keeping only the first occurrence.

            Args:
                lane: A list of points representing a lane.

            Returns:
                filtered_lane: A list of points representing a lane without points with the same Y coordinate.
        """
        assert lane[-1][1] <= lane[0][1] # Invalid lane
        filtered_lane = []
        used = set()
        for p in lane:
            if p[1] not in used:
                filtered_lane.append(p)
                used.add(p[1])

        return filtered_lane
    
    def __sample_lane(self, points:list) -> tuple:
        """
            Sample the lane points at the anchor points using the interpolation function and extrapolate the lane with a straight line.
            It also separates the points inside and outside the image.

            Args:
                points: A list of points representing a lane.
            
            Returns:    
                xs_outside_image: The x coordinates of the lane points outside the image.
                xs_inside_image: The x coordinates of the lane points inside the image.
        """
        points = np.array(points)
        x, y = points[:, 0], points[:, 1]

        # Interpolate points inside domain
        assert len(points) > 1
        interp = InterpolatedUnivariateSpline(y[::-1], x[::-1], k=min(3, len(points) - 1))
        domain_min_y = y.min()
        domain_max_y = y.max()
        # Get anchor points inside the domain of the labeled lane
        sample_ys_inside_domain = self.__ys[(self.__ys >= domain_min_y) & (self.__ys <= domain_max_y)]
        assert len(sample_ys_inside_domain) > 0
        # Evaluates the y samples in the interpolation function to get the x samples
        interp_xs = interp(sample_ys_inside_domain)

        # Extrapolate lane with a straight line using the 2 points closest to the bottom
        two_closest_points = points[:2]
        extrap = np.polyfit(two_closest_points[:, 1], two_closest_points[:, 0], deg=1)
        extrap_ys = self.__ys[self.__ys > domain_max_y]
        extrap_xs = np.polyval(extrap, extrap_ys)
        all_xs = np.hstack((extrap_xs, interp_xs))

        # Separate between inside and outside points
        inside_mask = (all_xs >= 0) & (all_xs < 1)
        xs_inside_image = all_xs[inside_mask]
        xs_outside_image = all_xs[~inside_mask]

        return xs_outside_image, xs_inside_image

    def __getitem__(self, idx):
        """
            Get an item from the dataset.

            Args:
                idx: The index of the item to get.

            Returns:
                A tuple containing the image (matrix) and the label (tensor of lane format).
        """
        item = self.annotations[idx]
        img_org = cv2.imread(item['path'])
        img_org = cv2.resize(img_org, (self.img_w, self.img_h))
        img = ToTensor()((img_org.copy()/255.0).astype(np.float32))
        label = item['label']
        return (img, label)
    
    def __len__(self):
        return len(self.annotations)
    

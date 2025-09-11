import xml.etree.ElementTree as ET
import cv2
import numpy as np
from numpy.f2py.auxfuncs import throw_error
from scipy.interpolate import UnivariateSpline
import json
import os

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def obtain_string_points(self):
        return f"({self.x},{self.y})"
    def obtain_tuple_point(self):
        return (int(self.x), int(self.y))

class Polyline:
    def __init__(self, points_string):
        if not points_string:
            return

        self.points_array = []
        self.miny = np.inf
        self.maxy = 0

        coords = points_string.split(';')

        for coord in coords:
            if coord.strip():
                x, y = map(float, coord.split(','))
                self.points_array.append(Point(x, y))

                self.miny = min(self.miny, y)
                self.maxy = max(self.maxy, y)

        self.func = self.fit_spline()

    def return_np_array(self):
        """Return numpy array formatted for cv2.polylines()"""
        if not self.points_array:
            return np.array([], dtype=np.int32).reshape(0, 2)

        # Create array of [x, y] coordinates
        points_list = [[curr_point.x, curr_point.y] for curr_point in self.points_array]

        # Convert to numpy array with int32 type (required by cv2.polylines)
        return np.array(points_list, dtype=np.int32)

    def fit_spline(self):
        if len(self.points_array) < 2:
            return lambda y: self.points_array[0].x if self.points_array else 0

            # Sort points by y-coordinate (top to bottom)
        sorted_points = sorted(self.points_array, key=lambda p: p.y)
        y_coords = [p.y for p in sorted_points]
        x_coords = [p.x for p in sorted_points]


        # Use light smoothing to handle annotation noise
        # s = number_of_points gives good balance between smoothness and accuracy
        spline = UnivariateSpline(y_coords, x_coords, s=len(y_coords) * 5)
        return lambda y: float(spline(y))

    def get_x_coordinates(self, h_samples):
        """Get x coordinates for given y coordinates (h_samples)"""
        x_coords = []
        for y in h_samples:
            # Only add x coordinate if y is within the polyline's range
            if self.miny <= y <= self.maxy:
                x_coords.append(int(self.func(y)))
            else:
                x_coords.append(-2)  # Standard value for out-of-range
        return x_coords

def parse_images_from_xml(file_path):
    """
    Parse XML file and extract image data1 with polylines.

    Args:
        file_path (str): Path to the XML file

    Returns:
        List of dictionaries containing image data1
    """
    tree = ET.parse(file_path)
    root = tree.getroot()

    images = []

    # Find all image elements
    for image in root.findall('image'):
        image_data = {
            'id': image.get('id'),
            'name': image.get('name'),
            'width': image.get('width'),
            'height': image.get('height'),
            'polylines': []
        }

        # Find all polylines in this image
        for curr_polyline in image.findall('polyline'):
            polylineObject = Polyline(curr_polyline.get('points'))
            polyline_data = {
                'label': curr_polyline.get('label'),
                'source': curr_polyline.get('source'),
                'occluded': curr_polyline.get('occluded'),
                'object': polylineObject,
                'z_order': curr_polyline.get('z_order')
            }
            image_data['polylines'].append(polyline_data)

        images.append(image_data)

    return images


# Example usage
if __name__ == "__main__":
    # Parse the XML file
    MAIN_FOLDER = "test"
    H_SAMPLES = list(range(0, 720, 10))
    images = parse_images_from_xml(f'{MAIN_FOLDER}/annotations.xml')

    all_labels = []

    # Print the results
    for image in images:
        lanes = []
        print(f"Image: {image['name']} ({image['width']}x{image['height']})")
        print(f"ID: {image['id']}")

        img = cv2.imread(f"{MAIN_FOLDER}/images/{image['name']}")

        for i, polyline in enumerate(image['polylines']):
            polyline_obj = polyline['object']
            print(f"  Polyline {i + 1}:")
            print(f"    Label: {polyline['label']}")
            print(f"    Source: {polyline['source']}")
            print(f"    Points: {[p.obtain_string_points() for p in polyline_obj.points_array[:3]]}...")
            x_coords = polyline_obj.get_x_coordinates(H_SAMPLES)
            lanes.append(x_coords)

            if len(lanes) > 2:
                throw_error("error! wrong lane")
        #     for point in polyline_obj.points_array:
        #         cv2.circle(img, point.obtain_tuple_point(), 5, (255,0,0), -1)
            valid_points = [[x, y] for x, y in zip(x_coords, H_SAMPLES) if x != -2]
            valid_points = np.array(valid_points, dtype=np.int32)
            cv2.polylines(img, [valid_points], False, (255, 255, 255), 3)

        json_output = {
            "lanes": lanes,
            "h_samples": H_SAMPLES,
            "raw_file": image['name']
        }

        all_labels.append(json_output)
        print("-" * 50)

        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        cv2.imshow("Image", img)
        cv2.resizeWindow("Image", 640, 360)
        cv2.waitKey()

    with open(f'{MAIN_FOLDER}/labels.json', "w") as f:
        for label in all_labels:
            f.write(json.dumps(label) + "\n")

    print(f"Generated {len(all_labels)} labels in data/labels.json")

import json
import cv2
import numpy as np
import os
import torch
from laneatt.laneatt import LaneATT

# LaneATT Model Configuration
MODEL_TO_LOAD = 'laneatt_100.pt'
CONFIG_TO_LOAD = 'laneatt.yaml'
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', MODEL_TO_LOAD)
CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'configs', CONFIG_TO_LOAD)

__img_w = 640
__img_h = 360
__anchor_y_discretization = 72
device = 'cuda'

def obtain_lanes(output: torch.Tensor, image: np.ndarray):
    """
        Obtain the lane lines in a list
    """

    proposals_length = output[:, 4]
    ys = torch.linspace(__img_h, 0, __anchor_y_discretization, device=device).cpu().numpy()

    lanes = []
    img = cv2.resize(image, (__img_w, __img_h))

    for lane_idx, lane in enumerate(output):
        x_coords = lane[5:].cpu().detach().numpy()
        length = int(proposals_length[lane_idx].item())
        points = [[int(x_coords[i]), int(ys[i])] for i in range(length)]
        lanes.append(points)
        # print(points)

        # points_np = np.array(points, dtype=np.int32)
        # cv2.polylines(img, [points_np], False, (0, 255, 0), 2)

    # cv2.imshow("image", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return lanes


def prediction(img, laneatt):
    """
    Make prediction on an image and draw the predicted middle line
    """
    # Get original image dimensions
    orig_h, orig_w = img.shape[:2]

    # Perform inference
    output = laneatt.cv2_inference(img)
    output = laneatt.nms_v2(output)

    # Get lanes (coordinates are in 640x360 space)
    lanes = obtain_lanes(output, img)

    # Calculate scale factors
    scale_x = orig_w / __img_w
    scale_y = orig_h / __img_h

    # Calculate middle line if we have 2 lanes
    pred_slope = None
    if len(lanes) == 2:
        lane1 = np.array(lanes[0])
        lane2 = np.array(lanes[1])

        # Get overlapping y range
        y_min = max(lane1[:, 1].min(), lane2[:, 1].min())
        y_max = min(lane1[:, 1].max(), lane2[:, 1].max())

        # Fit lines for both lanes
        poly1 = np.poly1d(np.polyfit(lane1[:, 1], lane1[:, 0], 1))
        poly2 = np.poly1d(np.polyfit(lane2[:, 1], lane2[:, 0], 1))

        # Calculate middle points
        y_samples = np.linspace(y_min, y_max, 20)
        middle_x = (poly1(y_samples) + poly2(y_samples)) / 2

        # Fit middle line using LSM
        middle_coeffs = np.polyfit(y_samples, middle_x, 1)
        pred_slope = middle_coeffs[0]  # Store predicted slope
        middle_poly = np.poly1d(middle_coeffs)

        # Scale and draw
        x_start = int(middle_poly(y_min) * scale_x)
        x_end = int(middle_poly(y_max) * scale_x)
        y_start = int(y_min * scale_y)
        y_end = int(y_max * scale_y)

        cv2.line(img, (x_start, y_start), (x_end, y_end), (255, 0, 0), 10)

    return pred_slope


def calculate_yaw_angle_difference(gt_slope, pred_slope):
    """
    Calculate yaw angle difference between ground truth and prediction

    Args:
        gt_slope: slope of ground truth line (k in x = ky + b)
        pred_slope: slope of predicted line

    Returns:
        angle difference in degrees
    """
    if gt_slope is None or pred_slope is None:
        return None

    theta_gt = np.arctan(gt_slope)
    theta_pred = np.arctan(pred_slope)

    angle_diff = np.abs(theta_gt - theta_pred)
    angle_diff_degrees = np.degrees(angle_diff)

    return angle_diff_degrees

# Initialize LaneATT model
print("Loading LaneATT model...")
laneatt = LaneATT(CONFIG_PATH)
laneatt.load(MODEL_PATH)
laneatt.eval()
print("Model loaded successfully!")

# Path to the JSON file
json_path = "/home/joel/Documents/research/RealTime-LaneATT/dataset_generator/test/labels.json"

# Base directory for images (assuming images are relative to the JSON file location)
base_dir = os.path.dirname(json_path)

# Read the JSON file
with open(json_path, 'r') as f:
    lines = f.readlines()

# Process each line (each line is a separate JSON object)
sum = 0
n = 0
for line in lines:
    data = json.loads(line.strip())

    # Get the image path
    image_path = os.path.join(base_dir, data['raw_file'])

    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        continue

    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to read image: {image_path}")
        continue

    # Get lanes and h_samples
    lanes = data['lanes']
    h_samples = data['h_samples']

    # Collect all lane lines
    lane_lines = []
    for lane in lanes:
        # Collect valid points
        valid_points = []
        for x, y in zip(lane, h_samples):
            # Skip invalid points (where x == -2)
            if x != -2:
                valid_points.append((x, y))

        # Need at least 2 points to fit a line
        if len(valid_points) < 2:
            continue

        lane_lines.append(valid_points)

    # Add prediction (blue line)
    pred_slope = prediction(img, laneatt)


    # Check if we have exactly 2 lanes to calculate middle
    if len(lane_lines) == 2:
        # Get points from both lanes
        lane1_points = lane_lines[0]
        lane2_points = lane_lines[1]

        # Convert to numpy arrays for easier manipulation
        lane1_x = np.array([p[0] for p in lane1_points])
        lane1_y = np.array([p[1] for p in lane1_points])
        lane2_x = np.array([p[0] for p in lane2_points])
        lane2_y = np.array([p[1] for p in lane2_points])

        # Find common y values or interpolate
        # Get the overlapping y range
        y_min = max(np.min(lane1_y), np.min(lane2_y))
        y_max = min(np.max(lane1_y), np.max(lane2_y))

        # Fit lines for both lanes
        coeffs1 = np.polyfit(lane1_y, lane1_x, 1)
        poly1 = np.poly1d(coeffs1)
        coeffs2 = np.polyfit(lane2_y, lane2_x, 1)
        poly2 = np.poly1d(coeffs2)

        # Sample y values in the overlapping range
        y_samples = np.linspace(y_min, y_max, num=20)

        # Calculate x values for both lanes at these y positions
        x1_samples = poly1(y_samples)
        x2_samples = poly2(y_samples)

        # Calculate middle points
        middle_x = (x1_samples + x2_samples) / 2
        middle_y = y_samples

        # Fit a line using LSM on the middle points
        middle_coeffs = np.polyfit(middle_y, middle_x, 1)
        middle_poly = np.poly1d(middle_coeffs)
        gt_slope = middle_coeffs[0]

        # Generate middle line endpoints
        y_start = int(y_min)
        y_end = int(y_max)
        x_start = int(middle_poly(y_start))
        x_end = int(middle_poly(y_end))

        # Draw the ground truth middle line in red
        cv2.line(img, (x_start, y_start), (x_end, y_end), color=(0, 0, 255), thickness=10)


    # Display the resized image
    cv2.imshow('Lane Visualization - Red: Ground Truth, Blue: Prediction', cv2.resize(img, (640, 360)))

    if gt_slope is not None and pred_slope is not None:
        yaw_diff = calculate_yaw_angle_difference(gt_slope, pred_slope)
        print(f"Yaw angle difference: {yaw_diff:.2f}°")
        sum += yaw_diff
        n += 1

    # Wait for key press
    key = cv2.waitKey(0)

print(f'Average yaw diff: {sum / n}')
cv2.destroyAllWindows()

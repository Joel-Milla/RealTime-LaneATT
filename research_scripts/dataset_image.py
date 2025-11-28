import json
import cv2
import numpy as np
import os

# Path to the JSON file
json_path = "/home/joel/Documents/research/RealTime-LaneATT/dataset_generator/train/labels.json"

# Base directory for images (assuming images are relative to the JSON file location)
base_dir = os.path.dirname(json_path)

# Read the JSON file
with open(json_path, 'r') as f:
    lines = f.readlines()

# Process each line (each line is a separate JSON object)
for line in lines:
    data = json.loads(line.strip())

    # Get the image path
    image_path = os.path.join(base_dir, data['raw_file'])

    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        continue

    # Read the image to get dimensions
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to read image: {image_path}")
        continue

    # Create a black image with the same dimensions
    height, width = img.shape[:2]
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Get lanes and h_samples
    lanes = data['lanes']
    h_samples = data['h_samples']

    # Draw each lane
    for lane in lanes:
        # Collect valid points
        valid_points = []
        for x, y in zip(lane, h_samples):
            # Skip invalid points (where x == -2)
            if x != -2:
                valid_points.append((x, y))

        # Need at least 3 points to fit a polynomial
        if len(valid_points) < 3:
            continue

        # Separate x and y coordinates
        x_coords = np.array([p[0] for p in valid_points])
        y_coords = np.array([p[1] for p in valid_points])

        # Fit a polynomial (degree 2 or 3 works well for lanes)
        # We fit x as a function of y since lanes are more vertical
        poly_degree = min(3, len(valid_points) - 1)
        coefficients = np.polyfit(y_coords, x_coords, poly_degree)
        polynomial = np.poly1d(coefficients)

        # Generate smooth curve points
        y_min, y_max = int(np.min(y_coords)), int(np.max(y_coords))
        y_curve = np.linspace(y_min, y_max, num=100)
        x_curve = polynomial(y_curve)

        # Create points for polylines
        curve_points = np.array([[int(x), int(y)] for x, y in zip(x_curve, y_curve)], dtype=np.int32)

        # Draw the polynomial curve
        # White color in BGR format, thickness 3
        cv2.polylines(img, [curve_points], isClosed=False, color=(255, 255, 255), thickness=15)

    # Display the resized image
    cv2.imshow('Lane Visualization', cv2.resize(img, (640, 360)))

    # Wait for key press
    key = cv2.waitKey(0)

    #
# CVAT to TuSimple Converter

This folder contains tools to convert dataset annotations from CVAT format to TuSimple format, which is required for training the LaneATT model.

## Files

- **[converter.py](converter.py)**: Converts dataset annotations from CVAT format to TuSimple format. TuSimple is the standardized annotation format required to train the lane detection model.

- **[json_cleaner.py](json_cleaner.py)**: Normalizes lane annotation data by standardizing the vertical sample points (h_samples). Converts lane format by:
  1. Standardizing h_samples to a fixed range (0 to 710 with step 10)
  2. Padding lanes with -2 values at the beginning to align with new h_samples
  3. Padding the end with -2 values as needed
  4. Verifying the conversion maintains the integrity of all annotation points

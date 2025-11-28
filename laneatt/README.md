# LaneATT

Source code for LaneATT used for training and making predictions.

## Overview

LaneATT is a lane detection model that uses attention mechanisms to detect and localize lane lines in images. The model uses a ResNet backbone to extract features, generates anchor proposals, and applies an attention mechanism to refine predictions through attention-based feature fusion.

## Key Components

### `__init__(config: str)`
Initializes the LaneATT model by loading configuration, setting up the ResNet backbone, generating anchor proposals, creating the attention mechanism layer, and initializing classification and regression heads.

### `forward(x: torch.Tensor)`
Main forward pass that:
1. Extracts feature maps from the ResNet backbone
2. Extracts anchor features from feature maps
3. Applies the attention mechanism to compute attention-weighted features
4. Outputs classification scores and regression predictions for each anchor proposal

### `cv2_inference(frame: np.ndarray)`
Performs inference on OpenCV frames by resizing, converting to tensor, running the forward pass, and applying postprocessing.

### `postprocess(output: torch.Tensor)`
Filters proposals by removing those with confidence scores below a positive threshold to reduce low-confidence predictions.

### `nms()` / `nms_v1()` / `nms_v2()` / `nms_v3()`
Different implementations of Non-Maximum Suppression that:
1. Filter proposals by confidence threshold
2. Compute similarity between proposals using mean absolute error
3. Remove duplicate or highly similar lane detections
4. Return unique, high-confidence lane detections

### `train_model(resume: bool = False)`
Training loop that:
1. Sets up optimizer and learning rate scheduler
2. Iterates through epochs, computing forward pass and losses
3. Updates model weights via backpropagation
4. Saves checkpoints and evaluates on validation set periodically

### `eval_model()`
Evaluation on validation set that computes:
- Classification and regression losses
- Precision, recall, F1 score, and accuracy metrics
- Saves evaluation metrics for analysis

### `__loss(proposals_list, targets)`
Computes combined loss consisting of:
- Classification loss (focal loss) for lane/non-lane predictions
- Regression loss (smooth L1) for lane position offsets

### `__match_proposals_with_targets(proposals, targets)`
Matches anchor proposals with ground truth targets by:
1. Computing distances between proposals and targets
2. Classifying proposals as positive (matched), negative (unmatched), or invalid
3. Computing intersection masks for valid regression regions

### `plot(output: torch.Tensor, image: np.ndarray)`
Visualizes detected lane lines by drawing them on the input image with random colors for each detected lane.

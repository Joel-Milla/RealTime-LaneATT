from laneatt import LaneATT
import torch
import cv2
import os
import random
import numpy as np

MODEL_TO_LOAD = 'laneatt_100.pt' # Model name to load
CONFIG_TO_LOAD = 'laneatt.yaml' # Configuration file name to load
IMG_TO_LOAD = 'test_img.png' # Image name to load
MODEL_PATH = os.path.join(os.path.dirname(__file__),'..', 'checkpoints', MODEL_TO_LOAD) # Model path (In this case, the model is in the same directory as the script)
CONFIG_PATH = os.path.join(os.path.dirname(__file__),'..', 'configs', CONFIG_TO_LOAD) # Configuration file path (In this case, the configuration file is in the same directory as the script)
# IMG_PATH = os.path.join(os.path.dirname(__file__), IMG_TO_LOAD) # Image path (In this case, the image is in the same directory as the script)
IMG_PATH = "/home/joel/Documents/research/RealTime-LaneATT/dataset_generator/test/images/frame_0730.png"
__img_w = 640
__img_h = 360
__anchor_y_discretization = 72
device = 'cuda'

def nms_v2(output: torch.Tensor, nms_threshold: float = 40.0) -> torch.Tensor:
    """
        Apply non-maximum suppression to the proposals

        Args:
            output (torch.Tensor): Regression proposals
            nms_threshold (float): NMS threshold

        Returns:
            torch.Tensor: Good proposals NMS suppressed
    """
    # Filter proposals with confidence below the threshold and sort them by confidence
    good_proposals = output[output[:, 1] > 0.5]
    good_proposals = good_proposals[good_proposals[:, 3].argsort(descending=True)]
    # Verify if there are no proposals
    if len(good_proposals) == 0: return good_proposals

    # Create a mask to store the same line proposals
    good_proposals_mask = np.zeros((len(good_proposals), len(good_proposals)), dtype=bool)

    starts = good_proposals[:, 2] / 360 * 72
    ends = good_proposals[:, 2] + good_proposals[:, 4]
    # Iterate over the proposals to filter out proposals that do not overlap in the y axis

    starts_a = starts.unsqueeze(1)
    ends_a = ends.unsqueeze(1)
    starts_b = starts.unsqueeze(0)
    ends_b = (ends - 1).unsqueeze(0)

    intersect_starts = torch.maximum(starts_a, starts_b).int()
    intersect_ends = torch.minimum(torch.minimum(ends_a, ends_b),
                                   torch.tensor(72)).int()

    valid_mask = intersect_starts < intersect_ends

    valid_pairs = torch.where(valid_mask)
    for idx in range(len(valid_pairs[0])):
        i, j = valid_pairs[0][idx].item(), valid_pairs[1][idx].item()

        start_idx = intersect_starts[i, j].item()
        end_idx = intersect_ends[i, j].item()

        if start_idx < end_idx:
            segment_a = good_proposals[i, 5 + start_idx:end_idx]
            segment_b = good_proposals[j, 5 + start_idx:end_idx]

            error = torch.mean(torch.abs(segment_a - segment_b)).item()

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
            high_confidence_unique_line_indexes[i] = \
                good_proposals[unique_line_indexes[i]:unique_line_indexes[i + 1]][:, 1].argmax().item()

        # Add an offset to counteract for the list slicing
        high_confidence_unique_line_indexes[i] += unique_line_indexes[i]

    return good_proposals[unique_line_indexes]


def obtain_lanes (output: torch.Tensor, image: np.ndarray):
    """
        Plot the lane lines on the image

        Args:
            output (torch.Tensor): Regression proposals
            image (np.ndarray): Image
    """

    proposals_length = output[:, 4]  # Of all the rows, get the column index 4 (which is the 5th column)
    # Get the y discretization values
    ys = torch.linspace(__img_h, 0, __anchor_y_discretization,
                        device=device)  # cre   ates a range from 0 to __img_h, where you need to have '__anchor_y_discretization' total number values

    # Convert to numpy and create points for each lane
    ys_np = ys.cpu().numpy()
    lanes = []

    # Resize the image to the model's trained size
    img = cv2.resize(image, (__img_w, __img_h))

    for lane_idx, lane in enumerate(output):
        # Get x coordinates for this lane
        x_coords = lane[5:].cpu().detach().numpy()

        # Create (x,y) points - only use valid length
        length = int(proposals_length[lane_idx].item())
        points = np.array([[x_coords[i], ys_np[i]] for i in range(length)], dtype=np.int32)
        lanes.append(points)
        # cv2.imshow(image)
        # cv2.waitKey(0)

        # Draw this lane
        cv2.polylines(img, [points], False, (0, 255, 0), 2)

    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return np.array([lanes])


def obtain_lanes2(output: torch.Tensor, image: np.ndarray):
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
        print(points)

        points_np = np.array(points, dtype=np.int32)
        cv2.polylines(img, [points_np], False, (0, 255, 0), 2)

    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return lanes

if __name__ == '__main__':
    laneatt = LaneATT(CONFIG_PATH) # Creates the model based on a configuration file
    laneatt.load(MODEL_PATH) # Load the model weights
    laneatt.eval() # Set the model to evaluation mode

    img = cv2.imread(IMG_PATH) # Read the image
    output = laneatt.cv2_inference(img) # Perform inference on the image
    # cv2.imshow("image", img)
    # cv2.waitKey(0)
    output = nms_v2(output)
    laneatt.plot(output, img)
    points = obtain_lanes2(output, img)
    print(points)
    print(len(points))
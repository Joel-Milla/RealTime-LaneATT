# from laneatt import LaneATT
from laneatt.laneatt import LaneATT

import cv2
import os
import time
import numpy as np
import torch

MODEL_TO_LOAD = 'laneatt_100.pt' # Model name to load
CONFIG_TO_LOAD = 'laneatt.yaml' # Configuration file name to load
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', MODEL_TO_LOAD) # Model path (In this case, the model is in the same directory as the script)
CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'configs', CONFIG_TO_LOAD) # Configuration file path (In this case, the configuration file is in the same directory as the script)

__img_w = 640
__img_h = 360
__anchor_y_discretization = 72
device = 'cuda'

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
        # print(points)

        # points_np = np.array(points, dtype=np.int32)
        # cv2.polylines(img, [points_np], False, (0, 255, 0), 2)

    # cv2.imshow("image", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return lanes

if __name__ == '__main__':
    laneatt = LaneATT(CONFIG_PATH) # Creates the model based on a configuration file
    laneatt.load(MODEL_PATH) # Load the model weights
    laneatt.eval() # Set the model to evaluation mode
    different_nms = [(laneatt.nms_v1, "nms_v1"), (laneatt.nms_v2, "nms_v2"), (laneatt.nms_v3, "nms_v3")]

    index = 1
    nms_func = different_nms[index][0]
    name = different_nms[index][1]
    average = 0
    count = 0

    # Open the camera
    cap = cv2.VideoCapture("/home/joel/Documents/research/RealTime-LaneATT/realsense/videos/v1_9_12_25.avi")
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f'Frame rate of video: {fps}')
    frame_delay = int(1000 / fps)  # Convert to milliseconds

    size = (1280, 720)  # Get frame dimensions (do this after reading first frame)
    # out = cv2.VideoWriter("output_video.avi", cv2.VideoWriter_fourcc(*'MJPG'), 15, size)

    while True:
        ret, frame = cap.read() # Read a frame from the camera

        if ret:
            start = time.time() # Start the timer
            output = laneatt.cv2_inference(frame) # Perform inference on the frame
            output = nms_func(output)
            inference_time = time.time() - start
            laneatt.plot(output, frame)  # Plot the lanes onto the frame and show it
            points = obtain_lanes2(output, frame)

            if len(points) != 0:
                average += len(points[1]) if len(points) > 1 else len(points[0])
                count += 1
            # out.write(new_frame)
            # print(f'Inference time of {name}: ', inference_time) # Print the inference time
            # average += inference_time
            # count += 1

            print("Average is: ", str(average / count))
            # Wait for 'q' key to quit
            if cv2.waitKey(1) == ord('q'):
                break
            # cv2.waitKey(0)
        else:
            # If the frame cannot be read, break the loop
            print("Cannot receive frame")
            break

    cap.release() # Release the camera
    cv2.destroyAllWindows() # Close the window

    average /= count
    print(f'Average number of points: ', average)
    print('****************************')

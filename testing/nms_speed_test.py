# from laneatt import LaneATT
from laneatt.laneatt import LaneATT

import cv2
import os
import time

MODEL_TO_LOAD = 'laneatt_100.pt' # Model name to load
CONFIG_TO_LOAD = 'laneatt.yaml' # Configuration file name to load
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', MODEL_TO_LOAD) # Model path (In this case, the model is in the same directory as the script)
CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'configs', CONFIG_TO_LOAD) # Configuration file path (In this case, the configuration file is in the same directory as the script)

if __name__ == '__main__':
    laneatt = LaneATT(CONFIG_PATH) # Creates the model based on a configuration file
    laneatt.load(MODEL_PATH) # Load the model weights
    laneatt.eval() # Set the model to evaluation mode
    different_nms = [(laneatt.nms_v1, "nms_v1"), (laneatt.nms_v2, "nms_v2"), (laneatt.nms_v3, "nms_v3")]

    cap = cv2.VideoCapture("/home/joel/Documents/research/RealTime-LaneATT/realsense/videos/video2.avi") # Open the camera
    for nms_func, name in different_nms:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        average = 0
        count = 0
        while True:
            ret, frame = cap.read() # Read a frame from the camera

            if ret:
                start = time.time() # Start the timer
                output = laneatt.cv2_inference(frame) # Perform inference on the frame
                output = nms_func(output) # This filter runs on the CPU and is slow, for real-time applications, it is recommended to implement it on the GPU
                inference_time = time.time() - start
                print(f'Inference time of {name}: ', inference_time) # Print the inference time
                average += inference_time
                count += 1
                # laneatt.plot(output, frame) # Plot the lanes onto the frame and show it

                # Wait for 'q' key to quit
                # if cv2.waitKey(1) == ord('q'):
                #     break
            else:
                # If the frame cannot be read, break the loop
                print("Cannot receive frame")
                break

        print(f'Average inference time of {name}: ', average / count)
        print('****************************')
        # cap.release() # Release the camera
        # cv2.destroyAllWindows() # Close the window

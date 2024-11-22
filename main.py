from laneatt import LaneATT

import cv2
import os
import time
import numpy as np

MODEL_TO_LOAD = 'laneatt_100.pt' # Model name to load
MODEL_PATH = os.path.join(os.path.dirname(__file__), MODEL_TO_LOAD) # Model path (In this case, the model is in the same directory as the script)
    
if __name__ == '__main__':
    laneatt = LaneATT('laneatt.yaml') # Creates the model based on a configuration file
    laneatt.load(MODEL_PATH) # Load the model weights
    laneatt.eval() # Set the model to evaluation mode
    
    cap = cv2.VideoCapture(0) # Open the camera
    while True:
        ret, frame = cap.read() # Read a frame from the camera

        if ret:
            start = time.time() # Start the timer
            output = laneatt.cv2_inference(frame) # Perform inference on the frame
            # output = laneatt.nms(output) This filter runs on the CPU and is slow, for real-time applications, it is recommended to implement it on the GPU
            print('Inference time: ', time.time() - start) # Print the inference time
            laneatt.plot(output, frame) # Plot the lanes onto the frame and show it

            # Wait for 'q' key to quit
            if cv2.waitKey(1) == ord('q'):
                break
        else:
            print("Cannot receive frame")
            break

    cap.release() # Release the camera
    cv2.destroyAllWindows() # Close the window
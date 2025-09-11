from laneatt import LaneATT

import cv2
import os

MODEL_TO_LOAD = 'laneatt_100.pt' # Model name to load
CONFIG_TO_LOAD = 'laneatt.yaml' # Configuration file name to load
VID_TO_LOAD = 'jitomate-2.mp4' # Image name to load
MODEL_PATH = os.path.join(os.path.dirname(__file__),'..', 'checkpoints', MODEL_TO_LOAD) # Model path (In this case, the model is in the same directory as the script)
CONFIG_PATH = os.path.join(os.path.dirname(__file__),'..', 'configs', CONFIG_TO_LOAD) # Configuration file path (In this case, the configuration file is in the same directory as the script)
VID_PATH = os.path.join(os.path.dirname(__file__),'video1280×720', VID_TO_LOAD)

if __name__ == '__main__':
    laneatt = LaneATT(CONFIG_PATH) # Creates the model based on a configuration file
    laneatt.load(MODEL_PATH) # Load the model weights
    laneatt.eval() # Set the model to evaluation mode

    vidcap = cv2.VideoCapture(VID_PATH)
    success, image = vidcap.read()
    if success:
        print("[INFO] Success loading the video")

    while success:
        success, img = vidcap.read()

        output = laneatt.cv2_inference(img) # Perform inference on the image
        # output = laneatt.nms(output) This filter runs on the CPU and is slow, for real-time applications, it is recommended to implement it on the GPU
        laneatt.plot(output, img) # Plot the lanes onto the image and show it
        if cv2.waitKey(10) == 27:  # exit if Escape is hit
            break

    cv2.destroyAllWindows() # Close the window
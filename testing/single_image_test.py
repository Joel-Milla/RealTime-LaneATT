from laneatt import LaneATT

import cv2
import os

MODEL_TO_LOAD = 'laneatt_100.pt' # Model name to load
CONFIG_TO_LOAD = 'laneatt.yaml' # Configuration file name to load
IMG_TO_LOAD = 'test_img.png' # Image name to load
MODEL_PATH = os.path.join(os.path.dirname(__file__),'..', 'checkpoints', MODEL_TO_LOAD) # Model path (In this case, the model is in the same directory as the script)
CONFIG_PATH = os.path.join(os.path.dirname(__file__),'..', 'configs', CONFIG_TO_LOAD) # Configuration file path (In this case, the configuration file is in the same directory as the script)
IMG_PATH = os.path.join(os.path.dirname(__file__), IMG_TO_LOAD) # Image path (In this case, the image is in the same directory as the script)

if __name__ == '__main__':
    laneatt = LaneATT(CONFIG_PATH) # Creates the model based on a configuration file
    laneatt.load(MODEL_PATH) # Load the model weights
    laneatt.eval() # Set the model to evaluation mode

    img = cv2.imread(IMG_PATH) # Read the image
    output = laneatt.cv2_inference(img) # Perform inference on the image
    # output = laneatt.nms(output) This filter runs on the CPU and is slow, for real-time applications, it is recommended to implement it on the GPU
    laneatt.plot(output, img) # Plot the lanes onto the image and show it
    cv2.waitKey(0) # Wait for a key to close the window
    cv2.destroyAllWindows() # Close the window
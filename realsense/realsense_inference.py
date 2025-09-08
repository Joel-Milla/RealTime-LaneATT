import pyrealsense2 as rs
import cv2
import os
import time
import numpy as np
from laneatt import LaneATT

MODEL_TO_LOAD = 'laneatt_100.pt' # Model name to load
CONFIG_TO_LOAD = 'laneatt.yaml' # Configuration file name to load
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', MODEL_TO_LOAD) # Model path (In this case, the model is in the same directory as the script)
CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'configs', CONFIG_TO_LOAD) # Configuration file path (In this case, the configuration file is in the same directory as the script)

if __name__ == '__main__':
    laneatt = LaneATT(CONFIG_PATH)  # Creates the model based on a configuration file
    laneatt.load(MODEL_PATH)  # Load the model weights
    laneatt.eval()  # Set the model to evaluation mode

    # Configure depth and color streams
    pipeline = rs.pipeline()  # object that manages camera streaming
    config = rs.config()  # Used to specify what streams you want (i.e. depth, color)

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)  # gets the current connected camera
    pipeline_profile = config.resolve(pipeline_wrapper)  # gets the best profile for the current camera connected
    device = pipeline_profile.get_device()  # obtain the device
    device_product_line = str(
        device.get_info(rs.camera_info.product_line))  # give name of the connected device (i.e. D400)

    # Check if the current device has rgb camera
    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("Color sensor required")
        exit(0)

    # Color camera, 640x480, 8bit bgr, 30fps
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    # Using all the previous config, start streaming
    pipeline.start(config)

    while True:
        # Wait for a coherent pair of frames
        frames = pipeline.wait_for_frames()  # blocks until color have frames
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convert images to numpy arrays that opencv can understand
        color_image = np.asanyarray(color_frame.get_data())

        start = time.time() # Start the timer
        output = laneatt.cv2_inference(color_image) # Perform inference on the frame
        # output = laneatt.nms(output) This filter runs on the CPU and is slow, for real-time applications, it is recommended to implement it on the GPU
        print('Inference time: ', time.time() - start) # Print the inference time
        laneatt.plot(output, color_image) # Plot the lanes onto the frame and show it

        # Wait for 'q' key to quit
        if cv2.waitKey(1) == ord('q'):
            break

    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()  # Close the window
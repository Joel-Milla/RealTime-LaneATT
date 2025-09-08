import pyrealsense2 as rs
import cv2
import os
import time
import numpy as np
import argparse

GLOBAL_WIDTH = 1280
GLOBAL_HEIGHT = 720

def create_video_writer(file_name):
    video_name = file_name
    if not video_name.endswith(".avi"):
        video_name += ".avi"

    video_path = os.path.join("videos", video_name)

    # Video properties
    size = (GLOBAL_WIDTH, GLOBAL_HEIGHT) #widthxheight
    fps = 30.0
    new_video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    return new_video_writer

if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--name", required=True,
                    help="name of the video to save")
    args = vars(ap.parse_args())
    video_writer = create_video_writer(args["name"])

    MODEL_TO_LOAD = 'laneatt_100.pt' # Model name to load
    CONFIG_TO_LOAD = 'laneatt.yaml' # Configuration file name to load
    MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', MODEL_TO_LOAD) # Model path (In this case, the model is in the same directory as the script)
    CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'configs', CONFIG_TO_LOAD) # Configuration file path (In this case, the configuration file is in the same directory as the script)

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

    # Color camera, 8bit bgr, 30fps
    config.enable_stream(rs.stream.color, GLOBAL_WIDTH, GLOBAL_HEIGHT, rs.format.bgr8, 30)

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
        video_writer.write(color_image)

        # Wait for 'q' key to quit
        if cv2.waitKey(1) == ord('q'):
            break

    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()  # Close the window
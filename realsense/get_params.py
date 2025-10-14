import pyrealsense2 as rs
import numpy as np

# Configuration variables
WIDTH = 1280
HEIGHT = 720
FPS = 15
COLOR_FORMAT = rs.format.bgr8

# Initialize pipeline
pipeline = rs.pipeline()
config = rs.config()

# Enable only RGB stream
config.enable_stream(rs.stream.color, WIDTH, HEIGHT, COLOR_FORMAT, FPS)

# Start pipeline
profile = pipeline.start(config)

# Get RGB intrinsics only
color_profile = profile.get_stream(rs.stream.color)
color_intrinsics = color_profile.as_video_stream_profile().get_intrinsics()

# Print RGB intrinsics
print("RGB Camera Intrinsics:")
print(f"  fx: {color_intrinsics.fx}, fy: {color_intrinsics.fy}")
print(f"  cx: {color_intrinsics.ppx}, cy: {color_intrinsics.ppy}")
print(f"  Distortion: {color_intrinsics.coeffs}")

pipeline.stop()

'''
********************************
RGB Camera Intrinsics:
  fx: 919.8773803710938, fy: 920.2100219726562
  cx: 648.5521240234375, cy: 362.18756103515625
  Distortion: [0.0, 0.0, 0.0, 0.0, 0.0]

***********************************
With calibration ros node:
camera matrix
910.622747 0.000000 602.954797
0.000000 904.403457 373.546778
0.000000 0.000000 1.000000

distortion
0.065957 -0.156512 0.009464 -0.021096 0.000000

rectification
1.000000 0.000000 0.000000
0.000000 1.000000 0.000000
0.000000 0.000000 1.000000

projection
891.659327 0.000000 565.687608 0.000000
0.000000 923.911527 379.498314 0.000000
0.000000 0.000000 1.000000 0.000000

'''
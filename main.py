from laneatt import LaneATT

import cv2
import os
import torch
import random
from PIL import Image

MODEL_TO_LOAD = 'laneatt_70.pt'
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'checkpoints', MODEL_TO_LOAD)

if __name__ == '__main__':
    # Load the model
    laneatt = LaneATT(config=os.path.join(os.path.dirname(__file__), 'configs', 'laneatt.yaml'))
    laneatt.load(MODEL_PATH)
    laneatt.eval()

    # Display the result
    img = cv2.imread(os.path.join(os.path.dirname(__file__), '20.jpg'))
    img = cv2.resize(img, (640, 360))

    # Perform inference
    output = laneatt(torch.from_numpy(img).unsqueeze(0).permute(0, 3, 1, 2).float()).squeeze(0)
    processed_output = laneatt.postprocess(output, threshold=1.0)

    for i, lane in enumerate(processed_output):
        prev_x, prev_y = lane[0]
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        print(output[i][4].item())
        for j, (x, y) in enumerate(lane):
            if j == int(output[i][4].item()): break
            cv2.line(img, (int(prev_x), int(prev_y)), (int(x), int(y)), color, 2)
            prev_x, prev_y = x, y

    cv2.imshow('frame', img)
    cv2.waitKey(0)
    
    # # Start the webcam
    # cap = cv2.VideoCapture(0)

    # while True:
    #     # Capture frame-by-frame
    #     ret, frame = cap.read()

    #     # If a frame was returned, display it
    #     if ret:
    #         # Perform inference
    #         output = laneatt(torch.from_numpy(frame).unsqueeze(0).permute(0, 3, 1, 2).float())
    #         output = laneatt.postprocess(output, threshold=0.2)

    #         for lane in output:
    #             prev_x, prev_y = lane[0]
    #             color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    #             for x, y in lane:
    #                 cv2.line(frame, (int(prev_x), int(prev_y)), (int(x), int(y)), color, 2)
    #                 prev_x, prev_y = x, y

    #         cv2.imshow('frame', frame)

    #         if cv2.waitKey(1) == ord('q'):  # Press 'q' to quit
    #             break
    #     else:
    #         print("Cannot receive frame")
    #         break
    
    # # When everything done, release the capture
    # cap.release()
    # cv2.destroyAllWindows()
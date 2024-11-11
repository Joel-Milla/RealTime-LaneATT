from laneatt import LaneATT
from torchvision.transforms import ToTensor

import cv2
import os
import time

import numpy as np

MODEL_TO_LOAD = 'laneatt_300.pt'
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'checkpoints', MODEL_TO_LOAD)
    
if __name__ == '__main__':
    laneatt = LaneATT(config=os.path.join(os.path.dirname(__file__), 'configs', 'laneatt.yaml'))
    laneatt.load(MODEL_PATH)
    laneatt.eval()
    
    cap = cv2.VideoCapture(0)
    #start, end = time.time(), time.time()
    while True:
        #start = time.time()
        ret, frame = cap.read()

        if ret:
            # Resize frame to the model's trained size
            frame = cv2.resize(frame, (laneatt.img_w, laneatt.img_h))
            # Convert frame to tensor
            img_tensor = ToTensor()((frame.copy()/255.0).astype(np.float32)).permute(0, 1, 2)

            # Predict
            output = laneatt(img_tensor.unsqueeze(0)).squeeze(0)

            # Plot the lanes above the threshold onto the frame and show it
            laneatt.plot(output, frame, threshold=0.0)

            # Wait for 'q' key to quit
            if cv2.waitKey(1) == ord('q'):
                break
        else:
            print("Cannot receive frame")
            break

    cap.release()
    cv2.destroyAllWindows()
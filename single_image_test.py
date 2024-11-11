from laneatt import LaneATT
from torchvision.transforms import ToTensor

import cv2
import os
import yaml
import json

import numpy as np

MODEL_TO_LOAD = 'laneatt_300.pt'
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'checkpoints', MODEL_TO_LOAD)
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'configs', 'laneatt.yaml')
    
if __name__ == '__main__':
    laneatt = LaneATT(config=os.path.join(os.path.dirname(__file__), 'configs', 'laneatt.yaml'))
    laneatt.load(MODEL_PATH)
    laneatt.eval()
    
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    root = config['dataset']['test']['root']

    json_files = os.listdir(root)
    json_files = [f for f in json_files if f.endswith('.json')]

    for j in json_files:
        with open(os.path.join(root, j), 'r') as f:
            for line in f:
                line = json.loads(line)
                img_path = os.path.join(root, line['raw_file'])
                img = cv2.imread(img_path)
                img = cv2.resize(img, (laneatt.img_w, laneatt.img_h))
                img_tensor = ToTensor()((img.copy()/255.0).astype(np.float32)).permute(0, 1, 2)
                output = laneatt(img_tensor.unsqueeze(0)).squeeze(0)
                laneatt.plot(output, img, threshold=0.5)
                cv2.waitKey(0)
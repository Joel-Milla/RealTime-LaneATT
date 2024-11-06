from laneatt import LaneATT

import os

if __name__ == '__main__':
    laneatt = LaneATT(config=os.path.join(os.path.dirname(__file__), 'configs', 'laneatt.yaml'))
    laneatt.train_model(resume=True)
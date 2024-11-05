from laneatt import LaneATT

if __name__ == '__main__':
    laneatt = LaneATT(config='configs/laneatt.yaml')
    laneatt.train_model(resume=True)
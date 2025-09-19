from laneatt import LaneATT
import os

CONFIG_TO_LOAD = 'laneatt.yaml' # Configuration file name to load
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'configs', CONFIG_TO_LOAD) # Configuration file path (In this case, the configuration file is in the same directory as the script)

if __name__ == '__main__':
    laneatt = LaneATT(config=CONFIG_PATH) # Creates the model based on a configuration file
    laneatt.train_model() # Train the model
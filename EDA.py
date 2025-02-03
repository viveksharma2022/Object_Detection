"""
Exploratory data analysis of the object detection data from BDD100 dataset
"""
import numpy as np
import pandas as pd
import json
from utility import utils
from pathlib import Path

def read_parameter_file():
    """
    The function reads  a parameter file located in the same folder

    Returns:
        _dict_: dict containing key-value pairs read from json
    """
    return utils.data_loader\
            .parse_json_file(str(Path.cwd().joinpath("parameters.json")))

if __name__ == "__main__":

    params = read_parameter_file()

    data_loader = utils.yolo_data_loader()
    train_raw_labels = data_loader.parse_json_file(params["Train_Labels"])
    data_loader.format_labels(train_raw_labels)


    pause = 1
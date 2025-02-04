"""
Exploratory data analysis of the object detection data from BDD100 dataset
"""
import numpy as np
import pandas as pd
import json
from utility import utils
from pathlib import Path
import matplotlib.pyplot as plt

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

    #create directories
    params["fig_exports"] = Path.cwd().joinpath("exports/figures")
    params["fig_exports"].mkdir(parents = True, exist_ok = True)

    data_loader = utils.yolo_data_loader()
    train_raw_labels = data_loader.parse_json_file(params["Train_Labels"])
    data_loader.format_labels(train_raw_labels)

    # visualize data distribution
    class_counts = data_loader.data["class_names"].value_counts()
    class_counts_percent = class_counts / np.sum(class_counts)
    print("\n")
    print(f"Data samples per class \n {class_counts}")
    print("The class counts can indicate that majority of classes are cars and therefore this is an imabalanced dataset")
    print("\n")
    
    average_sizes_per_class = data_loader.data.groupby("class_names")[["area", "aspect_ratio"]].mean()
    print("\n")
    print(f"Average bounding box sizes per class \n {average_sizes_per_class}")
    print("\n")

    #plot and export figures
    plt.figure()
    plt.subplot(2,1,1)
    class_counts.plot(kind = "bar")
    plt.ylabel("Counts")
    plt.title("Class counts")
    plt.xticks(rotation=45)
    plt.subplot(2,1,2)
    class_counts_percent.plot(kind = "bar")
    plt.ylabel("Percent")
    plt.title("Class counts percent")
    plt.xticks(rotation=45)
    plt.tight_layout()
    exportPath = params["fig_exports"].joinpath("class_distribution.jpg")
    plt.savefig(exportPath)
    plt.close()

    plt.figure()
    plt.subplot(2,1,1)
    average_sizes_per_class["area"].plot(kind = "bar")
    plt.ylabel("in pixels")
    plt.title("Average size of bounding boxes")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.subplot(2,1,2)
    average_sizes_per_class["aspect_ratio"].plot(kind = "bar")
    plt.title("Average aspect ratio of bounding boxes")
    plt.xticks(rotation=45)
    plt.tight_layout()
    exportPath = params["fig_exports"].joinpath("average_size_distribution.jpg")
    plt.savefig(exportPath)
    plt.close()

    eda_stats = [class_counts, average_sizes_per_class]
    exportPath = Path.cwd().joinpath("exports/stats.txt")
    utils.export_pdSeries_to_txt(eda_stats, str(exportPath))

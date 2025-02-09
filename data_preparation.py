from utility import utils
from EDA import read_parameter_file 
from pathlib import Path

if __name__ == "__main__":
    """Prepares the dataset for yolo object detection model training and validation
    """

    params = read_parameter_file()

    # create folders to export 
    params["train_labels_yolo"]  = Path(params["Train_Images"]).parent.joinpath("labels")
    params["val_labels_yolo"]    = Path(params["Val_Images"]).parent.joinpath("labels")

    params["train_labels_yolo"].mkdir(parents = True, exist_ok = True)
    params["val_labels_yolo"].mkdir(parents = True, exist_ok = True)

    # export train labels
    data_loader_train = utils.yolo_data_loader()
    train_raw_labels = data_loader_train.parse_json_file(params["Train_Labels"])
    data_loader_train.format_labels(train_raw_labels)
    data_loader_train.export_labels(params["image_width"], params["image_height"], params["train_labels_yolo"])

    # export validation labels
    data_loader_val = utils.yolo_data_loader()
    val_raw_labels = data_loader_train.parse_json_file(params["Val_Labels"])
    data_loader_val.format_labels(val_raw_labels)
    data_loader_val.export_labels(params["image_width"], params["image_height"], params["val_labels_yolo"])
    
    
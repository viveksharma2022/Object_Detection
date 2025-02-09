import json
from abc import ABC, abstractmethod
import pandas as pd
from pathlib import Path
import logging

class_names = {
        'person': 0,
        'rider': 1,
        'car': 2,
        'truck': 3,
        'bus': 4,
        'train': 5,
        'motor': 6,
        'bike': 7,
        'traffic light': 8,
        'traffic sign':9
}

class data_loader(ABC):
    """_summary_
    interface for data loading
    """
    def __init__(self):
        self.version = "v0.0.0"

    @staticmethod
    def parse_json_file(file_name):
        with open(file_name, 'r') as f:
            return json.load(f)

    @abstractmethod
    def format_labels(self, input_labels):
        pass

class yolo_data_loader(data_loader):
    """
    Specific implementation for data loading for the yolo model

    """
    def __init__(self):
        super().__init__()
        self.data_labels = None

    def format_labels(self,input_labels):
        """
        Format the labels so as to extract relevant information for EDA and label export for ultralytics yolo

        Args:
            input_labels (list): resulting list from json parser
        """
        yolo_labels = []
        for entry in input_labels:
            for label in entry["labels"]:
                if label["category"] in class_names:
                    yolo_labels.append({"name": entry["name"],
                                        "class_id": class_names[label["category"]],
                                        "class_names": label["category"],
                                        "x1": label['box2d']['x1'],
                                        "y1": label['box2d']['y1'],
                                        "x2": label['box2d']['x2'],
                                        "y2": label['box2d']['y2'],
                                        "area": (label['box2d']['x2'] - label['box2d']['x1']) * (label['box2d']['y2'] - label['box2d']['y1']),
                                        "aspect_ratio": (label['box2d']['x2'] - label['box2d']['x1']) / (label['box2d']['y2'] - label['box2d']['y1'])})
                
        self.data = pd.DataFrame(yolo_labels)

    def export_labels(self, image_width, image_height, exportPath):
        """_summary_
            Exports the labels as txt files to be used for the ultralytics yolo model

        Args:
            image_width (int): image width
            image_height (int): image height
            exportPath (pathlib Path): path to the export folder
        """
        images = self.data.groupby("name")
        for idx, (image, group) in enumerate(images):
            print(f"{idx}/{len(images)} Exporting label for image: {image}")
            export_label_file = exportPath.joinpath(Path(image).stem + ".txt") # remove jpg extension and add txt
            with open(str(export_label_file), 'w') as f:
                for row in group.itertuples():
                    class_id = row.class_id
                    width = row.x2 - row.x1
                    height = row.y2 - row.y1 
                    center_x = row.x1 + width/2
                    center_y = row.y1 + height/2
                    f.write(' '.join(map(str, [class_id, 
                                               round(center_x/image_width,4), 
                                               round(center_y/image_height,4),
                                               round(width/image_width,4), 
                                               round(height/image_height,4)])))
                    f.write('\n')

            

def export_pdSeries_to_txt(series_list, exporPath):
    """
    helper function to export list of pandas series data to a txt file

    Args:
        series_list (_type_): list containing pandas series elements
        exporPath (_type_): string containg the txt path, assuming the parent directory is available
    """
    with open(exporPath, 'w') as f:
        for series in series_list:
            f.write(series.to_string() + '\n\n')


class Logger:
    def __init__(self, name="object_detection"):
        # Create a logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)  # Default level can be changed

        # Create a console handler for logging to the console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Define the log format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)

        # Add the handlers to the logger
        self.logger.addHandler(console_handler)

    def get_logger(self):
        return self.logger
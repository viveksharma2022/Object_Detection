import json
from abc import ABC, abstractmethod
import pandas as pd

class_names = {
        'pedestrian': 0,
        'rider': 1,
        'car': 2,
        'truck': 3,
        'bus': 4,
        'train': 5,
        'motorcycle': 6,
        'bicycle': 7,
        'traffic light': 8,
        'traffic sign':9
}

class data_loader(ABC):
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
    def __init__(self):
        super().__init__()
        self.data_labels = None

    def format_labels(self,input_labels):
        yolo_labels = []
        for entry in input_labels:
            for label in entry["labels"]:
                if label["category"] in class_names:
                    yolo_labels.append({"name": entry["name"],
                                        "class_id": class_names[label["category"]],
                                        "x1": label['box2d']['x1'],
                                        "y1": label['box2d']['y1'],
                                        "x2": label['box2d']['x2'],
                                        "y2": label['box2d']['y2']})
                
        self.data = pd.DataFrame(yolo_labels)
            

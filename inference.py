import cv2
from EDA import read_parameter_file
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime

if __name__ == "__main__":

    params = read_parameter_file()
    model = YOLO(params["model_inference"])

    expPath = Path.cwd().joinpath("Inference_results//" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    expPath.mkdir(parents=True, exist_ok= True)
    
    for imgFile in Path(params["infer_Images"]).glob('*.jpg'):
        img = cv2.imread(imgFile)
        try:
            img = cv2.cvtColor(img)
        except Exception as e:
            print("Image is probably 3 channel already, no explicit conversion needed")
        
        result = model(img, conf = 0.5)
        result[0].save(expPath.joinpath(imgFile.name))


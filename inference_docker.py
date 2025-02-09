import cv2
import requests
import numpy as np
from pathlib import Path
from EDA import read_parameter_file
import json
from datetime import datetime
from utility.utils import Logger

def save_response_graphic(result: json, image: cv2.Mat, expPath: Path):
    """Draws the prediction boxes from the json output of model api and exports the file

    Args:
        result (json): result of the fast api inference response from the docker container
        image (cv2.Mat): inference image
        expPath (Path): path to save the image
    """
    class_names_colors = {
        'person': (185, 72, 94),
        'rider': (45, 218, 156),
        'car': (255, 255, 255),
        'truck': (67, 153, 210),
        'bus': (17, 184, 88),
        'train': (143, 96, 43),
        'motor': (8, 121, 168),
        'bike': (255, 195, 72),
        'traffic light': (199, 38, 142),
        'traffic sign':(70, 31, 194)
}
    
    predictions = json.loads(result.json())

    # Loop through predictions and draw bounding boxes
    for pred in predictions:
        # Extract bounding box and class name
        xmin, ymin, xmax, ymax = int(pred['box']['x1']), int(pred['box']['y1']), int(pred['box']['x2']), int(pred['box']['y2'])
        class_name = pred['name']
        confidence = pred['confidence']

        # Draw rectangle (bounding box)
        color = class_names_colors[pred['name']]
        thickness = 2
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, thickness)

        # Label text
        label = f"{class_name} {confidence:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.9
        font_thickness = 2

        # Get the size of the text
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)

        # Draw a filled rectangle for the background
        cv2.rectangle(image, (xmin, ymin - text_height - baseline),
                    (xmin + text_width, ymin),
                    color, cv2.FILLED)

        cv2.putText(image, label, (xmin, ymin - 10), font, font_scale, (0,0,0), font_thickness, cv2.LINE_AA)

    cv2.imwrite(expPath, image)

if __name__ == "__main__":
    
    logger = Logger().get_logger()

    # URL of your FastAPI app (adjust this if necessary)
    url = "http://localhost:8000/predict/"

    params = read_parameter_file()

    # path to export the inference result
    expPath = Path.cwd().joinpath("Inference_results//" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    expPath.mkdir(parents=True, exist_ok= True)

    # Convert the image (OpenCV Mat) to a byte array
    for imgFile in Path(params["infer_Images"]).glob('*.jpg'):
        logger.info(f"Inferencing: {imgFile}")
        # Load an image with OpenCV
        image = cv2.imread(imgFile)

        try:
            image = cv2.cvtColor(image)
        except Exception as e:
            logger.warn("Image is probably 3 channel already, no explicit conversion needed")

        _, img_encoded = cv2.imencode('.jpg', image)  # Encoding in JPG format
        img_bytes = img_encoded.tobytes()  # Convert the encoded image to byte stream

        # Send the image as a POST request with the 'files' parameter
        files = {'file': ('image.jpg', img_bytes, 'image/jpeg')}  # Name the file and set content type
        result = requests.post(url, files=files)

        if (result.status_code != 200): 
            logger.error(f"API response status code: {result.status_code}")
        else: # SUCCESS:
            exportPath = expPath.joinpath(imgFile.name)
            save_response_graphic(result, image, exportPath)
            logger.info(f"Saving result: {exportPath}")
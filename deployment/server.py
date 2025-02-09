import numpy as np
from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import cv2
 
model = YOLO("deployment/yolov5_best.pt")

class_names = np.array(['person',
                        'rider',
                        'car',
                        'truck',
                        'bus',
                        'train',
                        'motor',
                        'bike',
                        'traffic light',
                        'traffic sign'])

app = FastAPI()

@app.get('/')
def reed_root():
    return{"message": 'Object detection on BDD_dataset yolov5'}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read the uploaded image as bytes
    image_bytes = await file.read()
    
    # Convert the byte data into a NumPy array
    np_array = np.frombuffer(image_bytes, np.uint8)
    
    # Decode the NumPy array into an OpenCV image (cv::Mat)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    # Run inference on the image using YOLO
    results = model(image, conf = 0.5)

    return results[0].to_json()



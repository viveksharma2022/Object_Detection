import torch
from ultralytics import YOLO

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"GPU devices available: {torch.cuda.device_count()}")

    model = YOLO("yolov5n.pt")
    print(model)

    freeze_layers = 10 # freeze the backbone

    # Freeze Backbone, for transfer learning
    freeze = [f"model.{x}." for x in range(freeze_layers)]  # layers to freeze
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            print(f"freezing {k}")
            v.requires_grad = False

    print("\n All layers")
    for name, param in model.named_parameters():
        print(name)

    results = model.train(data='./yolo_data.yaml', epochs=50, imgsz=640, device=0)
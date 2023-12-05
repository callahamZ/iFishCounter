# def latihModel():
from ultralytics import YOLO
# from IPython import display
# display.clear_output()

# yolo task=detect mode=train epochs=50 data=data.yaml model=yolov8s.pt imgsz=640 batch=6

# import ultralytics
# ultralytics.checks()
# Load a model
model = YOLO('yolov8s.pt')  # build a new model from YAML

# Train the model
dataPath = "D:\ILHAM\CODES\PYCHARM PROJECT\iFishCounter WMK\Data Classes\dataikan3\data.yaml"
results = model.train(data=dataPath, epochs=50, batch=8)

# if __name__ == '__main__':
# latihModel()
print("Training sukses")
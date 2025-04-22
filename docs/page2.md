# Code used to export the model in different versions

Below i will be listing the code used in the project to export the model in different formats, when we train a yolo model we get it in a `.pt` format which is a native pytorch format of the model.

## Code used to export the model to onnx

``` py title="onnx_converter.py" linenums="1"
from ultralytics import YOLO

# Load a model
model = YOLO(r"yolov8n.pt")  # load an official model
model = YOLO(r"E:\Aryabhatta_motors_computer_vision\scripts\models\best.pt")  # load a custom trained model

# Export the model
model.export(format="onnx")
```

This code uses the native yolo library called ultralytics to export the yolo model into onnx.

## Code used to export the yolo model to openvino

```py title="yolo_openvino_export.py" linenums="1"

from ultralytics import YOLO  # For YOLOv5 and YOLOv8

model = YOLO(r"D:\Aryabhatta_computer_vision\Yolov8_custom\scripts\models\best(1).pt")

model.export(format="openvino")

```

The above code is used to export the yolo model into the openvino format which is a quantised version of the model.
# Home
This documentation is a guide and a explaination to all the scripts used in the pothole tracking project.

## Directory structure of the project

The structure of the project is given below

```
├───datasets
│   └───coco8
│       ├───images
│       │   ├───train
│       │   └───val
│       └───labels
│           ├───train
│           └───val
├───docs
├───Object_detection
│   └───build
│       ├───CMakeFiles
│       │   ├───3.30.0-rc4
│       │   │   ├───CompilerIdC
│       │   │   │   └───Debug
│       │   │   │       └───CompilerIdC.tlog
│       │   │   ├───CompilerIdCXX
│       │   │   │   └───Debug
│       │   │   │       └───CompilerIdCXX.tlog
│       │   │   └───VCTargetsPath
│       │   │       └───x64
│       │   │           └───Debug
│       │   │               └───VCTargetsPath.tlog
│       │   └───84b16d11d6c0300ba713afeda4afe298
│       ├───Debug
│       ├───Object_detection.dir
│       │   └───Debug
│       │       └───Object_detection.tlog
│       └───x64
│           └───Debug
│               ├───ALL_BUILD
│               │   └───ALL_BUILD.tlog
│               └───ZERO_CHECK
│                   └───ZERO_CHECK.tlog
├───Pothole_Tracking_Comprehensive
│   ├───datasets
│   │   └───coco8
│   │       ├───images
│   │       │   ├───train
│   │       │   └───val
│   │       └───labels
│   │           ├───train
│   │           └───val
│   ├───docs
│   ├───Object_detection
│   │   └───build
│   │       ├───CMakeFiles
│   │       │   ├───3.30.0-rc4
│   │       │   │   ├───CompilerIdC
│   │       │   │   │   └───Debug
│   │       │   │   │       └───CompilerIdC.tlog
│   │       │   │   ├───CompilerIdCXX
│   │       │   │   │   └───Debug
│   │       │   │   │       └───CompilerIdCXX.tlog
│   │       │   │   └───VCTargetsPath
│   │       │   │       └───x64
│   │       │   │           └───Debug
│   │       │   │               └───VCTargetsPath.tlog
│   │       │   └───84b16d11d6c0300ba713afeda4afe298
│   │       ├───Debug
│   │       ├───Object_detection.dir
│   │       │   └───Debug
│   │       │       └───Object_detection.tlog
│   │       └───x64
│   │           └───Debug
│   │               ├───ALL_BUILD
│   │               │   └───ALL_BUILD.tlog
│   │               └───ZERO_CHECK
│   │                   └───ZERO_CHECK.tlog
│   ├───runs
│   │   └───detect
│   │       └───train
│   │           └───weights
│   ├───scripts
│   │   ├───Camera_calibration_images
│   │   ├───models
│   │   │   └───best(1)_openvino_model
│   │   ├───Split
│   │   │   ├───Split1
│   │   │   └───Split2
│   │   ├───videos_images
│   │   ├───yolov8_test
│   │   └───__pycache__
│   ├───train
│   │   ├───images
│   │   └───labels
│   ├───val
│   │   ├───images
│   │   └───labels
│   └───YOLOv8
│       └───yolov8n
│           └───weights
├───runs
│   └───detect
│       └───train
│           └───weights
├───scripts
│   ├───Camera_calibration_images
│   ├───models
│   │   └───best(1)_openvino_model
│   ├───Split
│   │   ├───Split1
│   │   └───Split2
│   ├───videos_images
│   ├───yolov8_test
│   └───__pycache__
├───train
│   ├───images
│   └───labels
├───val
│   ├───images
│   └───labels
└───YOLOv8
    └───yolov8n
        └───weights
```
In this documentation most of our focus will be on what the scripts in the `./scripts` directory do.
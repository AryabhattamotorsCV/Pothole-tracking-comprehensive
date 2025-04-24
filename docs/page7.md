# Code used for debugging

When implementing a system like this, there will be a lot of errors. I will be explaining code I used to solve those problems.


## Code used to find the index of the camera

When using multiple cameras knowing the index of the camera is important, to know this we run loops and detect the index at which the camera is available.

```py title="camera_index_finder.py" linenums="1"

import cv2

cams_test = 500
for i in range(0, cams_test):
    cap = cv2.VideoCapture(i)
    test, frame = cap.read()
    if test:
        print("i : "+str(i)+" /// result: "+str(test))
```
 The code is self explanatory.


## Code used to test the servo motors

When implementing the system we need to bring the servo motors to a particular position and we also need to check if the motor works before running the main code, we use code given below to do that.

```py title="Pyfirmata_test_code_motor_running.py" linenums="1"

import pyfirmata
from pyfirmata import Arduino, SERVO, util
from time import sleep
port = 'COM7' #usb pin
pin = 10 #pin which servo is connected to on digital
pin1 = 9
board = pyfirmata.Arduino(port)
board.digital[pin].mode = SERVO
board.digital[pin1].mode = SERVO    

def rotate_servo(pin,angle):
     board.digital[pin].write(angle)
     board.digital[pin1].write(angle)
     sleep(0.0015)


user_angle_1 = int(input("Input user angle1: "))
user_angle_2 = int(input("Input user angle2: "))

rotate_servo(pin, user_angle_1)
rotate_servo(pin1, user_angle_2)

```

The library pyfirmata is used here to rotate the servo motor.


## Undistorting the camera video stream

Camera have some inherent distortion and to fix this we need to undistort the image. Though I didn't use this feature in my system, It might be useful in case you need to use it.

```py title="" linenums="1"

import cv2
import numpy as np

# Load your camera matrix and distortion coefficients
# You need to calibrate your camera beforehand to get these values
mtx = np.array([[892.86025081, 0, 319.72816385],
                   [0, 894.36852612, 182.8589494 ],
                      [0, 0, 1]])
dist = np.array([-2.17954003e-02, 3.78637782e-01, 4.81147467e-04, 1.20099914e-03,
  -8.01539551e-01])  # Replace with actual values

# Open webcam feed
cap = cv2.VideoCapture(1)

# Get frame dimensions
ret, frame = cap.read()
h, w = frame.shape[:2]

# Compute the optimal new camera matrix and undistort maps
new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, new_camera_mtx, (w, h), 5)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Undistort the frame
    undistorted_frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
    
    # Display the result
    cv2.imshow("distorted_frame", frame)
    cv2.imshow("Undistorted Feed", undistorted_frame)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
```

## Code to check if the model is readable

When running code in one of the office laptops I faced an issue of model path being read as null, Below I have given code to see if the model path is readable or not.

```py title="model_path_checker.py" linenums="1"

import os

file_path = r"D:\Arybhatta_motors_computer_vision\Yolov8_custom\scripts\models\best-yolov11n.pt"
if os.path.exists(file_path):
    print("File exists!")
else:
    print("File does not exist.")

```

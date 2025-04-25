# Script used to develop and evaluate the tracking system and mechanism

This section details the code used for the tracking mechanism in a lab setting. It's important to note that there are two approaches for implementing the tracking system: one involves mounting the camera, while the other involves fixing the camera in place.

## Code used to implement the solution with a mounted camera

The below code is used to implement the mounted camera method, it uses a tensorflow ssd model to run inference.

```py linenums="1" title="tensorflow_ssd_object_detection_OPENCV.py"

# --- Import necessary libraries ---
import os
import cv2
import numpy as np
import sys
import glob
import random
import importlib.util
from tensorflow.lite.python.interpreter import Interpreter
import matplotlib.pyplot as plt
import time

# --- Import PyFirmata for Arduino control ---
import pyfirmata
from pyfirmata import Arduino, SERVO, util
from time import sleep

# --- Set up Arduino board and define servo pins ---
port = 'COM6'
pin = 10   # Horizontal movement servo
pin1 = 9   # Vertical movement servo
board = pyfirmata.Arduino(port)
board.digital[pin].mode = SERVO
board.digital[pin1].mode = SERVO

# --- Function to rotate a servo motor connected to the given pin ---
def rotate_servo(pin, angle):
    board.digital[pin].write(angle)

# Initialize servo angles
angle = 45 
angle1 = 45
rotate_servo(pin, angle)
rotate_servo(pin1, angle1)

# --- Load TensorFlow Lite model ---
modelpath = r"D:\Aryabhatta_computer_vision\Yolov8_custom\scripts\models\detect1.tflite"
interpreter = Interpreter(model_path=modelpath)
interpreter.allocate_tensors()
min_conf = 0.55  # Minimum confidence threshold
arr = []  # Store inference times

# --- Get model input/output details ---
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
float_input = (input_details[0]['dtype'] == np.float32)
input_mean = 127.5
input_std = 127.5

# --- Load label map ---
with open("labelmap.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# --- Start capturing video ---
video_path = r"D:\Aryabhatta_computer_vision\Yolov8_custom\scripts\videos\WhatsApp Video 2024-07-05 at 01.41.50_72d4a5c5 (online-video-cutter.com).mp4"
cap = cv2.VideoCapture(1)  # Use webcam (change to 0 or path if needed)

# --- Main loop for processing video frames ---
while cap.isOpened():
    success, frame = cap.read()
    if frame is None:
        break

    if success:
        image = cv2.flip(frame, 1)  # Flip frame horizontally for mirror effect
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        imH, imW, _ = image.shape
        print(image.shape)

        # Resize frame to model's expected input size
        image_resized = cv2.resize(image_rgb, (width, height))
        input_data = np.expand_dims(image_resized, axis=0)

        if float_input:
            input_data = (np.float32(input_data) - input_mean) / input_std

        # --- Run inference ---
        start_time = time.time()
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        end_time = time.time()

        # Calculate and store inference time
        total_time = end_time - start_time
        arr.append(total_time)
        FPS = str(int(1 / (total_time)))  # Frames per second

        # --- Get model outputs ---
        boxes = interpreter.get_tensor(output_details[1]['index'])[0]    # Bounding boxes
        classes = interpreter.get_tensor(output_details[3]['index'])[0]  # Class indices
        scores = interpreter.get_tensor(output_details[0]['index'])[0]   # Confidence scores

        detections = []

        # --- Loop through detections ---
        for i in range(len(scores)):
            if ((scores[i] > min_conf) and (scores[i] <= 1.0)):
                # Convert bounding box coordinates to image size
                ymin = int(max(1, (boxes[i][0] * imH)))
                xmin = int(max(1, (boxes[i][1] * imW)))
                ymax = int(min(imH, (boxes[i][2] * imH)))
                xmax = int(min(imW, (boxes[i][3] * imW)))
                x_center = (xmin + xmax) / 2
                y_center = (ymin + ymax) / 2

                print(x_center, y_center)

                # Draw bounding box
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)

                # Deadband region is defined
                if (x_center > 300 and x_center < 420) and (y_center > 200 and y_center < 320):
                    continue

                # Servo logic: move left/right based on x position
                if x_center < 320:
                    angle = angle - 1
                    rotate_servo(pin, angle)
                if x_center > 320:
                    angle = angle + 1
                    rotate_servo(pin, angle)

                # Servo logic: move up/down based on y position
                if y_center < 240:
                    angle1 = angle1 - 1
                    rotate_servo(pin1, angle1)
                if y_center > 240:
                    angle1 = angle1 + 1
                    rotate_servo(pin1, angle1)

                # Prepare and draw label
                object_name = labels[int(classes[i])]
                label = '%s: %d%%' % (object_name, int(scores[i] * 100))
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                label_ymin = max(ymin, labelSize[1] + 10)
                cv2.rectangle(image, (xmin, label_ymin - labelSize[1] - 10),
                              (xmin + labelSize[0], label_ymin + baseLine - 10), (255, 255, 255), cv2.FILLED)
                cv2.putText(image, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

                # Append detection details
                detections.append([object_name, scores[i], xmin, ymin, xmax, ymax])

        # Display FPS on frame
        cv2.putText(image, f"FPS = {FPS}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (178, 255, 102), 2)
        
        # Show the annotated frame
        cv2.imshow('annotated frame', image)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
```
The code above adjusts the position of the mechanism by comparing the center of the bounding box of the pothole with the center of the camera. It corrects the position by calculating the error in each iteration and gradually reducing it. However, the issue with this approach is that the algorithm may cause the mechanism to overshoot its target.

## Code used to implement the stationary camera solution

```py linenums="1" title="yolo_object_detection_image_plus_depth_est.py"

import cv2
import numpy as np 
from ultralytics import YOLO
from Equation_solver_closed_form import tangent_point_finder # User defined function to calculate the coordinates
import math
from Pyfirmata_servo_motor_runner import rotate_servo # User defined function to rotate the servo at an angle.


WIDTH_HALF_PPRGN = 40          # Half of the width of the perspective transform region
LENGTH_VERTICAL = 57.5         # The verical width of the perspective transform region

motor2_pin = 10                # Servo motor pin for horizontal axis
motor1_pin = 6                 # Servo motor pin for vertical axis

# Coordinates for perspective transformation (4 corner points)
tl = [51, 44]                  # Top-left
bl = [56, 279]                 # Bottom-left
tr = [476, 37]                 # Top-right
br = [628, 225]                # Bottom-right

# Fixed hardware-related values
distance = 72                  # Distance from the mechanism to the base line of the perspetive transform region
Radius = 3.319                 # Radius of the circular path of the laser (unit matches coordinate system)
H = 77                         # Height of the mechanism from the ground

# ---------------------- Camera and Model Setup ---------------------- #
image_path = r"D:\Aryabhatta_computer_vision\Yolov8_custom\scripts\Depth_sample.jpg"

# Load YOLOv8 model
model = YOLO(r"D:\Aryabhatta_computer_vision\Yolov8_custom\scripts\models\best.pt")

# Start video capture
cam_port = 0
cam = cv2.VideoCapture(cam_port)

if not cam.isOpened():
    print("Error could not load camera") 

# ---------------------- Main Processing Loop ---------------------- #
while cam.isOpened():
    result, image_0 = cam.read()   
    if result:
        image = cv2.flip(image_0, 1)                   # Flip the image horizontally for mirror view
        imH, imW, _ = image.shape                     # Get image dimensions
        print(image.shape)

        # Resize the image and draw perspective reference points
        image_resized = cv2.resize(image, (640, 480))
        for point in [tl, bl, tr, br]:
            cv2.circle(image_resized, point, 5, (0, 0, 255), -1)

        # Apply perspective transform
        pts1 = np.float32([tl, bl, tr, br])
        pts2 = np.float32([[0, 0], [0, 480], [640, 0], [640, 480]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        result_0 = cv2.warpPerspective(image_resized, matrix, (640, 480))

        # ---------------------- YOLO Inference ---------------------- #
        result = model.predict(result_0, save=True, project="./", name="yolov8_test", exist_ok=True, conf=0.6)
        boxes = result[0].boxes                       # Extract bounding boxes from detection result

        color = (255, 248, 150)                       # Box color

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]              # Coordinates of detected object
            print("Box coordinates:", x1, y1, x2, y2)

            # Calculate center of bounding box
            x_center = float((x1 + x2) / 2)
            y_center = float((y1 + y2) / 2)    

            # Draw bounding box
            image_resized = cv2.rectangle(result_0, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=4)

            # Convert pixel coordinates to real-world coordinates (in cm)
            x_coordinate = (79 / 640) * x_center              
            y_coordinate = (58 / 480) * y_center 

            # Translate coordinates to be relative to camera origin
            x_coordinate_trfm = WIDTH_HALF_PPRGN - x_coordinate  
            y_coordinate_trfm = distance + (LENGTH_VERTICAL - y_coordinate)

            print(x_coordinate_trfm, y_coordinate_trfm)

            # ---------------------- Compute Tangent Point ---------------------- #
            Tangent_points = tangent_point_finder(x_coordinate_trfm, y_coordinate_trfm, Radius)
            print(Tangent_points)

            # Shift origin for angle calculations
            origin_shift_x = x_coordinate_trfm - Tangent_points[0]
            origin_shift_y = y_coordinate_trfm - Tangent_points[1] 

            # Calculate servo angles
            servo_motor1_angle_radians = math.atan(H / math.sqrt(origin_shift_x**2 + origin_shift_y**2))
            servo_motor2_angle_radians = math.atan(Tangent_points[1] / Tangent_points[0])
            servo_motor1_angle_degrees = math.degrees(servo_motor1_angle_radians)
            servo_motor2_angle_degrees = math.degrees(servo_motor2_angle_radians)

            # ---------------------- Actuate Servos Based on Object Position ---------------------- #
            if x_coordinate_trfm > 0:  # Object is in right quadrant
                print(x_coordinate_trfm)
                motor2_angle = 180 - abs(servo_motor2_angle_degrees) + 5
                motor1_angle = servo_motor1_angle_degrees 
                print(f"Motor 2 angle = {motor2_angle} Motor 1 angle = {motor1_angle}")
                print(f'Horizontal distance in cm is {x_coordinate}, Vertical distance in cm is {y_coordinate}')
                print("right quadrant")
                rotate_servo(motor2_pin, motor2_angle)
                rotate_servo(motor1_pin, motor1_angle)

            if x_coordinate_trfm < 0:  # Object is in left quadrant
                print(x_coordinate_trfm)
                motor2_angle = servo_motor2_angle_degrees
                motor1_angle = 180 - servo_motor1_angle_degrees
                print(f"Motor 2 angle = {motor2_angle} Motor 1 angle = {motor1_angle}")
                print(f'Horizontal distance in cm is {x_coordinate}, Vertical distance in cm is {y_coordinate}')
                print("left quadrant")
                rotate_servo(motor2_pin, motor2_angle)
                rotate_servo(motor1_pin, motor1_angle)

        # Save annotated image for inspection
        cv2.imwrite("test_image.jpg", image_resized)
        cv2.imshow("yolov8_testing", result_0)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break    

# ---------------------- Cleanup ---------------------- #
cap.release()
cv2.destroyAllWindows()   
```
The above code uses specific mathematical methods to calculate the coordinates of the servo motor, a process known as inverse kinematics.

## PID control system for the mounted camera setup

PID is a control method that allows us to manage various aspects of motor motion. Below is the code for implementing the PID control system.

```py title="PID_camera_tracking.py" linenums="1"

# Import required libraries
import cv2
import numpy as np
import time
import serial
from simple_pid import PID
from ultralytics import YOLO  # YOLO from Ultralytics
import torch

# Select the CUDA device (GPU)
torch.cuda.set_device(0)

# Load YOLOv11n model - replace with your trained model path
model = YOLO(r"D:\Aryabhatta_computer_vision\Yolov8_custom\scripts\models\best-yolov11n.pt")
print("before: ", model.device.type)

# Test run on a sample image to ensure the model loads correctly
results = model(r"E:\Aryabhatta_motors_computer_vision\images_potholes\78778.png")
print("after: ", model.device.type)

# Setup serial communication with Arduino (adjust COM port accordingly)
arduino = serial.Serial(port='COM7', baudrate=9600, timeout=1)
time.sleep(2)  # Give some time to establish the serial connection

# Initialize PID controllers for X and Y servos (horizontal and vertical)
pid_x = PID(0.02, 0, 0, setpoint=0)
pid_y = PID(0.02, 0, 0, setpoint=0)

# Set servo angle limits
pid_x.output_limits = (-45, 45)
pid_y.output_limits = (-45, 45)

# Deadband (tolerance) in pixels - no correction if within this range
deadband_x = 10
deadband_y = 15

# Start video capture (adjust the index based on camera setup)
cap = cv2.VideoCapture(1)

# Initial angles for the servo motors
servo_angle_x = 90  # Mid position for horizontal
servo_angle_y = 40  # Some default vertical position

def send_servo_command(servo, angle):
    """ Send servo angle command to Arduino over serial """
    command = f"{servo}:{int(angle)}\n"
    arduino.write(command.encode())
    time.sleep(0.02)  # Small delay for serial communication stability

# Send the initial servo position
send_servo_command("X", servo_angle_x)
send_servo_command("Y", servo_angle_y)

try:
    while True:
        # Read a frame from the camera and horizontally flip it
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        if not ret:
            break

        # Run tracking with confidence threshold
        results = model.track(frame, persist=True, conf=0.5)

        # Set box color for annotation
        color = (255, 248, 150)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Extract bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                object_center_x = (x1 + x2) / 2
                object_center_y = (y1 + y2) / 2

                # Calculate center of the frame
                frame_center_x = frame.shape[1] // 2
                frame_center_y = frame.shape[0] // 2

                # Compute X and Y errors from center
                error_x = object_center_x - frame_center_x
                error_y = object_center_y - frame_center_y

                # If horizontal error exceeds tolerance, correct with PID
                if abs(error_x) > deadband_x:
                    error_x = error_x.cpu().numpy()
                    correction_x = pid_x(error_x)
                    servo_angle_x = np.clip(servo_angle_x - correction_x, 0, 180)
                    print(servo_angle_x)
                    send_servo_command("X", servo_angle_x)

                # If vertical error exceeds tolerance, correct with PID
                if abs(error_y) > deadband_y:
                    error_y = error_y.cpu().numpy()
                    correction_y = pid_y(error_y)
                    servo_angle_y = np.clip(servo_angle_y - correction_y, 0, 60)
                    print(servo_angle_y)
                    send_servo_command("Y", servo_angle_y)

        # Display the annotated output
        annotated_frame = results[0].plot()
        cv2.imshow("Tracking", annotated_frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# On exit, release camera and serial resources
finally:
    cap.release()
    cv2.destroyAllWindows()
    arduino.close()
```

The deadband refers to the range where the mechanism won't move further if it's pointing within it. PID is a system aimed at minimizing error and determining the best way to reduce it, considering factors like speed, smoothness, and stability. The P (proportional) component controls how quickly the error is reduced, the I (integral) component addresses accumulated error over time, enhancing pothole detection accuracy, and the D (derivative) component stabilizes the motion, preventing overshooting. In our system, a high P value is necessary to match the speed at which the bike moves.


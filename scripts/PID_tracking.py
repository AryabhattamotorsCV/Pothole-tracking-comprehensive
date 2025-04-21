import cv2
import numpy as np
import time
from pyfirmata import Arduino, SERVO
from simple_pid import PID

# Arduino setup
board = Arduino('COM7')  # Adjust for your port (e.g., 'COM3' on Windows)
servo_pin = 9  # Adjust based on your setup
board.digital[servo_pin].mode = SERVO

# PID controller
pid = PID(0.15, 0.002, 0.02, setpoint=0)
pid.output_limits = (-90, 90)  # Servo movement range

# Start camera
cap = cv2.VideoCapture(0)
tracker = cv2.TrackerCSRT_create()  # Object tracker

# Select ROI for tracking
ret, frame = cap.read()
bbox = cv2.selectROI("Tracking", frame, False)
tracker.init(frame, bbox)

servo_angle = 90  # Initial servo angle
# servo.write(servo_angle)

time.sleep(1)


try:
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        if not ret:
            break

        success, bbox = tracker.update(frame)
        if success:
            x, y, w, h = [int(v) for v in bbox]
            object_center = x + w // 2
            frame_center = frame.shape[1] // 2
            error = object_center - frame_center
            correction = pid(error)
            print(correction)
            # Convert correction to servo angle
            correction = np.clip(correction, -90, 90)
            # servo.write(int(servo_angle))
            print(servo_angle + correction)
            # Draw tracking box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    board.exit()

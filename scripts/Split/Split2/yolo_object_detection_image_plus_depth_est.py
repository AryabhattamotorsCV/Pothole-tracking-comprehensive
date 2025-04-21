import cv2
import numpy as np 
from ultralytics import YOLO
from Equation_solver_closed_form import tangent_point_finder
import math
# from Pyfirmata_servo_motor_runner import rotate_servo
KNOWN_DISTANCE = 50
KNOWN_WIDTH = 14.5
face_width_in_frame = 238

motor2_pin = 10
motor1_pin = 6

tl = [51, 44]
bl = [56, 279]
tr = [476, 37]
br = [628, 225]
    

distance = 72
Radius = 3.319
H = 77    

image_path = r"D:\Aryabhatta_computer_vision\Yolov8_custom\scripts\Depth_sample.jpg"
# Load model
model = YOLO(r"D:\Aryabhatta_computer_vision\Yolov8_custom\scripts\models\best.pt")
#capture image
cam_port = 0
cam = cv2.VideoCapture(cam_port) 
  
# reading the input using the camera
if not cam.isOpened():
    print("Error could not load camera") 

while cam.isOpened():
    result, image_0 = cam.read()   
    if result:
        image = cv2.flip(image_0, 1)
        # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        imH, imW, _ = image.shape
        print(image.shape)
        image_resized = cv2.resize(image, (640, 480))
        cv2.circle(image_resized, tl, 5, (0,0,255,-1))
        cv2.circle(image_resized, bl, 5, (0,0,255,-1))
        cv2.circle(image_resized, tr, 5, (0,0,255,-1))
        cv2.circle(image_resized, br, 5, (0,0,255,-1))

        pts1 = np.float32([tl, bl, tr, br])
        pts2 = np.float32([[0,0], [0,480], [640,0], [640,480]])

        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        result_0 = cv2.warpPerspective(image_resized, matrix, (640,480))
    # Run inference on an image
        result = model.predict(result_0, save = True, project = "./", name = "yolov8_test" , exist_ok = True, conf = 0.6)

        boxes = result[0].boxes

        color = (255, 248, 150) 

        for box in boxes:
            print(box.xyxy[0])
            x1, y1, x2, y2 = box.xyxy[0]  # Get coordinates in format [x1, y1, x2, y2]
            print("Box coordinates:", x1, y1, x2, y2)
            x_center = float((x1+x2)/2)
            y_center = float((y1+y2)/2)    
            image_resized = cv2.rectangle(result_0, (int(x1),int(y1)), (int(x2),int(y2)), color , thickness = 4)
            x_coordinate = (79/640)*x_center             
            y_coordinate = (58/480)*y_center 
            x_coordinate_trfm = 40 - x_coordinate  # add correction to get corodinates from the camera origin
            y_coordinate_trfm = distance + (57.5 - y_coordinate) 
            print(x_coordinate_trfm, y_coordinate_trfm)
            Tangent_points = tangent_point_finder(x_coordinate_trfm, y_coordinate_trfm, Radius)
            print(Tangent_points)
            origin_shift_x = x_coordinate_trfm - Tangent_points[0]
            origin_shift_y = y_coordinate_trfm - Tangent_points[1] 
            servo_motor1_angle_radians = math.atan(H/math.sqrt(origin_shift_x**2 + origin_shift_y**2))
            servo_motor2_angle_radians = math.atan(Tangent_points[1]/Tangent_points[0])
            servo_motor1_angle_degrees = math.degrees(servo_motor1_angle_radians)
            servo_motor2_angle_degrees = math.degrees(servo_motor2_angle_radians)
            if x_coordinate_trfm > 0:
                print(x_coordinate_trfm)
                motor2_angle = 180 - abs(servo_motor2_angle_degrees) + 5
                motor1_angle = servo_motor1_angle_degrees 
                print(f"Motor 2 angle = {motor2_angle} Motor 1 angle = {motor1_angle}")
                print(f'Horizontal distance in cm is {x_coordinate},Vertical distance in cm is {y_coordinate}')
                print("right quadrant")
                # rotate_servo(motor2_pin, motor2_angle)
                # rotate_servo(motor1_pin, motor1_angle)
            if x_coordinate_trfm < 0:
                print(x_coordinate_trfm)
                motor2_angle = servo_motor2_angle_degrees
                motor1_angle = 180 - servo_motor1_angle_degrees
                print(f"Motor 2 angle = {motor2_angle} Motor 1 angle = {motor1_angle}")
                print(f'Horizontal distance in cm is {x_coordinate},Vertical distance in cm is {y_coordinate}')
                print("left quadrant")
                # rotate_servo(motor2_pin, motor2_angle)
                # rotate_servo(motor1_pin, motor1_angle)
        # cv2.imwrite("test_image.jpg",image_resized)
        cv2.imshow("yolov8_testing" , result_0)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break    
        
cap.release()
cv2.destroyAllWindows()        

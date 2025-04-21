import cv2

from ultralytics import YOLO

import torch

import time

# torch.cuda.set_device(0)

# Load the YOLOv8 model
arr = []
model = YOLO(r"D:\Aryabhatta_computer_vision\Yolov8_custom\scripts\models\best1.pt")
# print("before: ",model.device.type)
results = model(r"E:\Aryabhatta_motors_computer_vision\images_potholes\78778.png")
# Open the video file
# print("after: ",model.device.type)
video_path = r"D:\Aryabhatta_computer_vision\Yolov8_custom\scripts\videos\WhatsApp Video 2024-07-05 at 01.41.50_72d4a5c5.mp4"
cap = cv2.VideoCapture(1)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    if frame is None:
        break
    # frame = cv2.resize(frame, (384, 640))
   
    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        start_time = time.time()
        results = model.track(frame, persist=True)
        end_time = time.time()
        inference_time = end_time - start_time
        FPS = str(int(1/inference_time))
        arr.append(inference_time)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]  # Get coordinates in format [x1, y1, x2, y2]
                print("Box coordinates:", x1, y1, x2, y2)
        # Visualize the results on the frame
            annotated_frame = results[0].plot()
        # Display the annotated frame
            # x_center = (x1+x2)/2
            # y_center = (y1+y2)/2
            # print(x_center, y_center)
                # if ((ymax-ymin)*(xmax-xmin)) > 50000:
                #     continue 
        if (x_center > 320 and x_center < 325) and (y_center > 240 and y_center < 245):
            continue
        if x_center < 320:
            angle = angle - 2
            rotate_servo(pin, angle)
        if x_center > 320:
            angle = angle + 2
            rotate_servo(pin, angle)
        if y_center < 240:
            angle1 = angle1 - 2
            rotate_servo(pin1, angle1)  
        if y_center > 240:
            angle1 = angle1 + 2
            rotate_servo(pin1, angle1)        
        cv2.putText(annotated_frame,f"FPS = {FPS}",(50, 50),cv2.FONT_HERSHEY_SIMPLEX, 1,(178,255,102), 2)
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
# print("average time =", sum(arr)/len(arr))
cap.release()
cv2.destroyAllWindows()
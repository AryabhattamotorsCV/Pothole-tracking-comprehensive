import cv2 
import numpy as np 
 
# Turn on Laptop's webcam

tl = [550, 338]
bl = [190, 487]
tr = [830, 338]  
br = [1164, 487]
url = "http://192.0.0.4:8080/video"
cap = cv2.VideoCapture(url)

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Pixel position: ({x}, {y})")

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    cv2.circle(frame, tl, 5, (0,0,255,-1))
    cv2.circle(frame, bl, 5, (0,0,255,-1))
    cv2.circle(frame, tr, 5, (0,0,255,-1))
    cv2.circle(frame, br, 5, (0,0,255,-1))
    cv2.circle(frame, (373, 316), 5, (0,0,255,-1))

    pts1 = np.float32([tl, bl, tr, br])
    pts2 = np.float32([[0,0], [0,480], [640,0], [640,480]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(frame, matrix, (640,480))
     
    # Wrap the transformed image
    cv2.imshow('frame', frame) # Initial Capture
    cv2.imshow('frame1', result) # Transformed Capture
    cv2.setMouseCallback("frame", click_event)
 
    if cv2.waitKey(24) == 27:
        break
 
cap.release()
cv2.destroyAllWindows()
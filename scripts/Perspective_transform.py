import cv2 
import numpy as np 
 
# Turn on Laptop's webcam

tl = [224, 68]
bl = [5, 223]
tr = [638, 149]  
br = [565, 420]
url = "http://192.0.0.4:8080/video"
cap = cv2.VideoCapture(3)

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Pixel position: ({x}, {y})")

while True:
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
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
    cv2.imshow('frame1', frame) # Transformed Capture
    cv2.setMouseCallback("frame", click_event)
 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
cap.release()
cv2.destroyAllWindows()
# Code used to implment and test the perspective transform

In this section I will be explaining the code used to find and test the perspective transform.

## Code used to find the coordinates of the corner of the flat surface

The below code is used to find the corner coordinates of a flat rectangular surface, this is done in order to apply perspective transform properly onto a flat surface when calculating the coordinates of the pothole.

```py title="Perspective_transform_coord_finder.py" linenums="1"
import cv2 
import numpy as np 
 
# === Turn on Laptop's webcam ===
# Index '3' selects the specific webcam device; change if needed (0 is usually default).
cap = cv2.VideoCapture(3)

# === Define four corner points in the original camera frame for perspective transformation ===
# These are manually chosen and represent a quadrilateral area that will be transformed to a rectangle.
tl = [224, 68]     # Top-left corner
bl = [5, 223]      # Bottom-left corner
tr = [638, 149]    # Top-right corner
br = [565, 420]    # Bottom-right corner

# === Callback function to capture mouse clicks ===
# Prints the pixel coordinates where the user clicks on the image window.
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Pixel position: ({x}, {y})")

# === Main loop to continuously capture frames from the webcam ===
while True:
    ret, frame = cap.read()  # Capture a single frame

    # Display the raw frame before any processing
    cv2.imshow('frame', frame)

    # === Draw red circles on the perspective transformation corners for visual reference ===
    cv2.circle(frame, tl, 5, (0, 0, 255), -1)
    cv2.circle(frame, bl, 5, (0, 0, 255), -1)
    cv2.circle(frame, tr, 5, (0, 0, 255), -1)
    cv2.circle(frame, br, 5, (0, 0, 255), -1)

    # Draw another reference point (optional, might be a target or calibration marker)
    cv2.circle(frame, (373, 316), 5, (0, 0, 255), -1)

    # === Prepare points for perspective transformation ===
    pts1 = np.float32([tl, bl, tr, br])  # Source points (distorted quadrilateral)
    pts2 = np.float32([[0, 0], [0, 480], [640, 0], [640, 480]])  # Destination points (rectangle)

    # === Compute the perspective transformation matrix ===
    matrix = cv2.getPerspectiveTransform(pts1, pts2)

    # === Apply the perspective warp using the matrix ===
    result = cv2.warpPerspective(frame, matrix, (640, 480))

    # === Show the frame after transformation ===
    # `frame1` is still showing the original frame here due to the line below,
    # likely a typo — should be `cv2.imshow('frame1', result)` if intent is to show transformed image.
    cv2.imshow('frame1', frame) 

    # Enable mouse click tracking on the original frame window
    cv2.setMouseCallback("frame", click_event)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
# === Release camera and close all OpenCV windows ===
cap.release()
cv2.destroyAllWindows()

```
The above code takes mouse clicks and prints the pixel coordinate of the location of the click, this helps in knowing the pixel coordinates of the corner just by clicking the mouse onto those corners in the camera image.


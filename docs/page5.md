# Code used to implement and test the perspective transform

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

## Code used to implement a dynamic perspective transform

This code implements a system where the four corners of a rectangular region are identified and a perspective transform is applied on to that.

```py title="Dynamic_perspective_transform_rect.py" linenums="1"

import cv2
import numpy as np

# Initialize the camera stream
cap = cv2.VideoCapture(0)

# Edge case handling
if not cap.isOpened():
    print("Error: Could not open video.")

# Loop to read the camera stream
while cap.isOpened():
    ret, image = cap.read()
    if not ret:
        break

    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define black color range (adjust as needed)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 50])

    # Mask for black regions (detecting tape)
    mask = cv2.inRange(hsv, lower_black, upper_black)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour (assuming it's the tape)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Approximate to a polygon
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)

        # Ensure we have exactly 4 corners
        if len(approx) == 4:
            corners = approx.reshape(4, 2)  # Reshape to (4,2) array
            print("Detected corners:", corners)

            # Draw the corners on the image
            for (x, y) in corners:
                cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

        else:
            print(f"Detected {len(approx)} corners, refining...")

    cv2.imshow("Detected Corners", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
```

The above code is used to find a rectangular region defined by black tapes, After that the rectangular region is found using contour detection.
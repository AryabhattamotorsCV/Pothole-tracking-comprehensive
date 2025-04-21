import cv2 


cap = cv2.VideoCapture(r"E:\Aryabhatta_motors_computer_vision\scripts\videos\WhatsApp Video 2024-07-05 at 01.41.49_9b652ede.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Display the frame
    cv2.imshow('Frame', frame)
    
    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
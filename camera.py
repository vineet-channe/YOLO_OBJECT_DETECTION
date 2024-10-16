# Import the OpenCV library
import cv2

# Open the default camera (index 0) if you include other like external webcam you can switch (1)cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(0)
# Set the frame width to 640 pixels
# cap.set(3, 640)

# Set the frame height to 480 pixels
# cap.set(4, 480)

# Infinite loop to continuously capture frames from the camera
while True:
    # Read a frame from the camera
    ret, img = cap.read()

    # Display the captured frame in a window named "Webcam"
    cv2.imshow('Cam', img)

    # Wait for a key press for 1 millisecond
    # If the pressed key is 'q', exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Release the camera
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
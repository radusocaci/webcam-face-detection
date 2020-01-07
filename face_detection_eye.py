""" Face detection program.
    This program detects the face/eyes in real time, using the webcam.
    The project is based on OpenCV. """
import cv2

""" Load classifiers used for face and eye detection. 
    A cascade is trained from positive and negative images (faces and non-faces). """
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Initialize webcam and start recording
cap = cv2.VideoCapture(0)

# Loop until ESC key was pressed
while True:

    # Read frames from camera one by one
    ret, img = cap.read()

    # Convert each frame to grayscale (most computations in OpenCV are done in grayscale)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in frame
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
    )

    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)

        # Compute face image (gray and normal)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

        # Detects eyes inside the encapsulating face rectangle
        eyes = eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=3
        )

        # Draw rectangle around eyes
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 127, 255), 2)

    # Display the resulting image (with face and eyes in rectangles)
    cv2.imshow('Video', img)

    # Wait for Esc key to stop
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

# Close the window
cap.release()

# De-allocate any associated memory usage
cv2.destroyAllWindows()

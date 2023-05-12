import cv2
# Loading the cascade for face detection
face_cascade = cv2.CascadeClassifier('C:/Users/utkar/AppData/Local/Programs/Python/Python310/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
# Loading the cascade for mask detection
mask_cascade = cv2.CascadeClassifier('C:/Users/utkar/AppData/Local/Programs/Python/Python310/Lib/site-packages/cv2/data/haarcascade_mcs_mouth.xml')
# Defining the video capture object
cap = cv2.VideoCapture(0)
while True:
    # Capturing the current frame here
    ret, frame = cap.read()
    # Converting the frame to grayscale here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detecting faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Iterating over each face
    for (x, y, w, h) in faces:
        # Drawing a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # Create a region of interest for the mask detection
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        # Detect masks in the region of interest
        mask = face_cascade.detectMultiScale(roi_gray)
        # Iterate over each mask
        for (mx, my, mw, mh) in mask:
            # Draw a rectangle around the mask
            cv2.rectangle(roi_color, (mx, my), (mx+mw, my+mh), (0, 0, 255), 2)
        
        if len(mask) == 0:
            cv2.putText(frame,'Mask On', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame,'No Mask', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    # Display the resulting frame
    cv2.imshow('Face Mask Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Release the video capture object
cap.release()
# Close all the windows
cv2.destroyAllWindows()
# preparing dataset with 5 set of emotion angry happyneutral sad and surprise
# traning model import library , use images and labels for training the model(traiing model keras and tensorflow
# face classifier file to detect face , emotion_detection.h5 for predicting emotion , test.py

import cv2
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array


face_classifier = cv2.CascadeClassifier('cascade.xml')
classifier = load_model('./Emotion_Detection.h5')

class_labels = ['angry', 'happy', 'neutral', 'sad', 'surprise']

# capturing video from webcam
cap = cv2.VideoCapture(0)

# Open a video file or start video capture from a camera
# cap = cv2.VideoCapture("video.mp4")

# Read and display video frames until the user presses a key
while cap.isOpened():
    # take a single frame from video
    ret, frame = cap.read()
    
    if not ret:
        break
    
    labels = []
    
    # converting into grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # used to perform face detection
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi = gray[y:y+h, x:x+w]
        
        if np.sum([roi]) != 0:
            roi = cv2.resize(roi, (48, 48))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            
            preds = classifier.predict(roi)[0]
            print("\nprediction = ", preds)
            label = class_labels[preds.argmax()]
            print("\nprediction max = ", preds.argmax())
            print("\nlabel = ", label)
            label_position = (x, y)
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        else:
            cv2.putText(frame, 'No Face Found', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
    
    cv2.imshow('Emotion Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

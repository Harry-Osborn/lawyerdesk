

import cv2
import dlib
from imutils import face_utils
from scipy.spatial import distance as dist
import numpy as np
from keras.models import load_model
from loggers.timer_decorator import timer_decorator


async def process_emotion(frame, face_classifier, classifier, detector):
    """
    Detects faces in a given frame, extracts the facial region of interest (ROI), resizes it, and predicts the emotion using a pre-trained classifier model.

    Args:
        - frame (numpy.ndarray): The input frame (image) in BGR format.
        - face_classifier (cv2.CascadeClassifier): A face classifier object used to detect faces in the frame.
        - classifier (keras.models.Model): A pre-trained emotion classifier model.
        - detector (dlib.fhog_object_detector): A face detector object used to ensure all detected faces have an emotion status.

    Returns:
        tuple: A tuple containing the annotated frame and a list of emotion statuses for each detected face.
    """
    emotion_labels = ['Angry', 'Disgust', 'Fear',
                      'Happy', 'Neutral', 'Sad', 'Surprise']
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected_faces = face_classifier.detectMultiScale(gray)
    emotion_statuses = []
    for (x, y, w, h) in detected_faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (224, 224),
                              interpolation=cv2.INTER_AREA)
        roi = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2RGB)
        roi = roi.astype('float') / 255.0
        roi = np.expand_dims(roi, axis=0)
        prediction = classifier.predict(roi)[0]
        emotion_status = emotion_labels[prediction.argmax()]
        emotion_statuses.append(emotion_status)
        label_position = (x, y)
        cv2.putText(frame, emotion_status, label_position,
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Ensure all detected faces have an emotion status
    faces = detector(gray)
    while len(emotion_statuses) < len(faces):
        emotion_statuses.append("Neutral")

    return frame, emotion_statuses

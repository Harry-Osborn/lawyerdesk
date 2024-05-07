

import cv2
import numpy as np
from loggers.timer_decorator import timer_decorator


async def process_mouth(frame, lm_model, detector):
    """
    Process the frame to detect faces and determine the status of the lips (open or closed) for each detected face.

    Args:
        - frame (numpy.ndarray): The input frame (image) to be processed.
        - lm_model (model): The landmark model used to detect facial landmarks.
        - detector (dlib.fhog_object_detector): The face detector used to detect faces in the frame.

    Returns:
        tuple: A tuple containing the processed frame and a list of lip statuses for each detected face.

    Example:
        - frame = cv2.imread('image.jpg')
        lm_model = load_model('landmark_model.h5')
        detector = dlib.get_frontal_face_detector()
        processed_frame, lip_statuses = process_mouth(frame, lm_model, detector)
        print(lip_statuses)
        # Output: ['Open', 'Closed', 'Open']
    """
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(img_gray)
    lip_statuses = []
    for face in faces:
        shapes = lm_model(img_gray, face)
        top_lip = [(shapes.part(i).x, shapes.part(i).y) for i in range(50, 53)]
        bottom_lip = [(shapes.part(i).x, shapes.part(i).y)
                      for i in range(56, 59)]
        top_mean = np.mean(top_lip, axis=0)
        bottom_mean = np.mean(bottom_lip, axis=0)
        d = abs(top_mean[1] - bottom_mean[1])
        lip_status = "Closed"
        if d > 20:
            lip_status = "Open"
        lip_statuses.append(lip_status)

    # Ensure all detected faces have a lip status
    while len(lip_statuses) < len(faces):
        lip_statuses.append("Closed")

    return frame, lip_statuses

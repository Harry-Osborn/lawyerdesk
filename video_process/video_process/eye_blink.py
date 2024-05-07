from loggers.timer_decorator import timer_decorator


import cv2
from imutils import face_utils
from scipy.spatial import distance as dist
import numpy as np


async def EAR_cal(eye):
    """
    Calculate the Eye Aspect Ratio (EAR) using the Euclidean distance formula.

    Args:
        - eye (list): A list of 6 2D coordinates representing the landmarks of an eye.

    Returns:
        - float: The Eye Aspect Ratio (EAR) value.
    """
    v1 = dist.euclidean(eye[1], eye[5])
    v2 = dist.euclidean(eye[2], eye[4])
    h1 = dist.euclidean(eye[0], eye[3])
    ear = (v1 + v2) / h1
    return ear


async def process_blink(frame, detector, lm_model, L_start, L_end, R_start, R_end):
    """
    Process the frame to detect faces, extract the landmarks of the eyes, calculate the Eye Aspect Ratio (EAR) for each eye,
    determine if the eyes are blinking or not based on a threshold, and return the frame with the blink statuses for each detected face.

    Args:
        - frame (numpy.ndarray): A numpy array representing an image frame.
        - detector: A face detector object that can detect faces in an image.
        - lm_model: A landmark model object that can extract landmarks from a face.
        - L_start (int): An integer representing the starting index of the left eye landmarks.
        - L_end (int): An integer representing the ending index of the left eye landmarks.
        - R_start (int): An integer representing the starting index of the right eye landmarks.
        - R_end (int): An integer representing the ending index of the right eye landmarks.

    Returns:
        tuple: A tuple containing the frame with the blink statuses for each detected face and a list of strings representing the blink statuses for each detected face.
    """
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(img_gray)
    blink_statuses = []
    for face in faces:
        shapes = lm_model(img_gray, face)
        shape = face_utils.shape_to_np(shapes)
        lefteye = shape[L_start:L_end]
        righteye = shape[R_start:R_end]
        left_EAR = await EAR_cal(lefteye)
        right_EAR = await EAR_cal(righteye)
        avg = (left_EAR + right_EAR) / 2
        blink_thresh = 0.5
        blink_status = "Not Blinking"
        if avg < blink_thresh:
            blink_status = "Blinking"
        blink_statuses.append(blink_status)

    # Ensure all detected faces have a blink status
    while len(blink_statuses) < len(faces):
        blink_statuses.append("Not Blinking")

    return frame, blink_statuses

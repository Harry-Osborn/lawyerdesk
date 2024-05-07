

import cv2
from scipy.spatial import distance as dist
import numpy as np


async def detect_gaze_direction(pupil_x, w):
    gaze_direction = "Center"
    if pupil_x < w * 0.3:
        gaze_direction = "Left"
    elif pupil_x > w * 0.6:
        gaze_direction = "Right"
    return gaze_direction


async def get_eye_region(eye_points, shape):
    eye_region = [(shape.part(point).x, shape.part(point).y)
                  for point in eye_points]
    return np.array(eye_region, np.int32)


async def process_gaze(frame, detector, predictor):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    gaze_directions = []
    for face in faces:
        shape = predictor(gray, face)
        left_eye_region = await get_eye_region(list(range(36, 42)), shape)
        right_eye_region = await get_eye_region(list(range(42, 48)), shape)
        gaze_status = "Center"
        for eye_region in [left_eye_region, right_eye_region]:
            eye_x = min(eye_region[:, 0])
            eye_y = min(eye_region[:, 1])
            eye_w = max(eye_region[:, 0]) - eye_x
            eye_h = max(eye_region[:, 1]) - eye_y
            roi_eye = gray[eye_y:eye_y + eye_h, eye_x:eye_x + eye_w]
            _, thresh_eye = cv2.threshold(
                roi_eye, 30, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(
                thresh_eye, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if len(contours) > 0:
                largest_contour = max(contours, key=cv2.contourArea)
                moments = cv2.moments(largest_contour)
                if moments["m00"] != 0:
                    cx = int(moments["m10"] / moments["m00"])
                    gaze_status = await detect_gaze_direction(cx, eye_w)
        gaze_directions.append(gaze_status)

    # Ensure all detected faces have a gaze direction
    while len(gaze_directions) < len(faces):
        gaze_directions.append("Center")

    return frame, gaze_directions

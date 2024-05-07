

import cv2
import mediapipe as mp
import numpy as np
import time
from loggers.timer_decorator import timer_decorator


import numpy as np
import cv2


async def calculate_angle(landmark1, landmark2, landmark3):
    """
    Calculate the angle between three points using the dot product and the arccosine function.

    Args:
        - landmark1 (list): The coordinates [x1, y1] of the first point.
        - landmark2 (list): The coordinates [x2, y2] of the second point (usually the joint).
        - landmark3 (list): The coordinates [x3, y3] of the third point.

    Returns:
        float: The angle between the three points in degrees.
    """
    a = np.array(landmark1) - np.array(landmark2)
    b = np.array(landmark3) - np.array(landmark2)
    angle = np.arccos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return np.degrees(angle)


async def body_posture(frame, pose, mp_drawing, mp_pose, previous_angles):
    """
    Perform pose estimation on the given frame and calculate the angles of the shoulders.

    Args:
        - frame (numpy.ndarray): The input frame (image or video frame) on which the pose estimation will be performed.
        - pose: The pose estimator object from the Mediapipe library.
        - mp_drawing: The drawing utility object from the Mediapipe library.
        - mp_pose: The pose module object from the Mediapipe library.
        - previous_angles (list): A list of previous shoulder angles.

    Returns:
        tuple: A tuple containing the annotated frame, movement message, and current angles.
            - annotated_frame (numpy.ndarray): The input frame with pose annotations.
            - movement_message (str): A message indicating if movement is detected.
            - current_angles (list): The angles of the left and right shoulders.
    """
    # Convert the frame to RGB (as mediapipe requires RGB images)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform pose estimation
    result = pose.process(frame_rgb)
    movement_message = ""

    # Draw the pose annotations on the frame
    if result.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        landmarks = result.pose_landmarks.landmark

        # Check if shoulders are visible
        shoulder_landmarks = [
            mp_pose.PoseLandmark.LEFT_SHOULDER,
            mp_pose.PoseLandmark.RIGHT_SHOULDER
        ]

        shoulders_visible = all(
            [landmarks[i].visibility > 0.5 for i in shoulder_landmarks]
        )

        if not shoulders_visible:
            cv2.putText(frame, "Please set up your camera", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            movement_message = "Please set up your camera"
            print(movement_message)
            return frame, movement_message, None
        else:
            # Extract coordinates
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y]
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP].y]

            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y]
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y]

            # Calculate angles
            left_shoulder_angle = await calculate_angle(left_elbow, left_shoulder, left_hip)
            right_shoulder_angle = await calculate_angle(right_elbow, right_shoulder, right_hip)

            current_angles = [left_shoulder_angle, right_shoulder_angle]

            if previous_angles:
                angle_differences = [abs(
                    current - previous) for current, previous in zip(current_angles, previous_angles)]

                if any(diff > 10 for diff in angle_differences):
                    movement_message = f"Movement detected: Left Shoulder Angle: {left_shoulder_angle:.2f}  Right Shoulder Angle: {right_shoulder_angle:.2f}"
                    print(movement_message)
            return frame, movement_message, current_angles
    else:
        cv2.putText(frame, "Please set up your camera", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        return frame, "Please set up your camera", None

# Sample usage
# frame, movement_message, current_angles = await body_posture(frame, pose, mp_drawing, mp_pose, previous_angles)

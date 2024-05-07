
import cv2
from loggers.timer_decorator import timer_decorator


async def headpose_process(frame, detector, predictor):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    direction = "Center"

    for face in faces:
        landmarks = predictor(gray, face)

        # Nose tip (landmark 34)
        nose = (landmarks.part(33).x, landmarks.part(33).y)
        cv2.circle(frame, nose, 2, (0, 255, 0), -1)

        # Head pose estimation (using nose and face rectangle)
        face_center = (face.left() + face.width() // 2,
                       face.top() + face.height() // 2)
        cv2.circle(frame, face_center, 2, (0, 0, 255), -1)

        x_diff = nose[0] - face_center[0]
        y_diff = nose[1] - face_center[1]

        direction = ""

        if abs(x_diff) > 20:  # Threshold to eliminate noise
            if x_diff > 0:
                direction += "Left "
            else:
                direction += "Right "

        if abs(y_diff) > 20:  # Threshold to eliminate noise
            if y_diff > 0:
                direction += "Down"
            else:
                direction += "Up"

        if direction == "":
            direction = "Center"

        cv2.putText(frame, direction, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame, direction

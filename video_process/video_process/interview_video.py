

import asyncio
import os
from modules.voice_bot.video_process.headpose import headpose_process
from modules.voice_bot.video_process.emotion import process_emotion
from modules.voice_bot.video_process.eye_blink import process_blink
from modules.voice_bot.video_process.gaze import process_gaze
from modules.voice_bot.video_process.mouth_blink import process_mouth
import cv2
import dlib
from imutils import face_utils
from datetime import datetime
from keras.models import load_model
from loggers.logger import log_event
from modules.voice_bot.video_process.detect_object import process_frame_with_yolo
from modules.voice_bot.video_process.body_postures import body_posture
import mediapipe as mp
from modules.voice_bot.video_process.detect_face import mark_attendance
from ultralytics import YOLO


class VideoProcessor:
    """
    The `VideoProcessor` class is responsible for processing video frames and extracting various features such as 
    blink status, gaze direction, mouth status, emotion, head pose, body postures, and object detection. 
    It uses multiple threads to parallelize the processing tasks and combines the results for visualization and 
    logging.
    """

    def __init__(self):
        """
        Initializes the VideoProcessor class by loading the required models and classifiers.
        """
        self.detector = dlib.get_frontal_face_detector()

        self.L_start, self.L_end = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        self.R_start, self.R_end = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        self.face_classifier = cv2.CascadeClassifier(
            r'loggers/saved_models/haarcascade_frontalface_default.xml')
        self.classifier = load_model(
            r'loggers/saved_models/Emotion_Detection.h5')
        self.lm_model = dlib.shape_predictor(
            r'loggers/saved_models/shape_predictor_68_face_landmarks.dat')
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose()
        self.previous_angles = None
        # Load the YOLOv8 model
        self.model = YOLO(r'loggers/saved_models/yolov8n.pt')

    async def video_processing(self, image_encoding, frame):
        """
        Processes a video frame by executing multiple processing tasks in parallel using ThreadPoolExecutor. Combines the results for visualization and logging.

        Args:
            - frame: The video frame to be processed.
            - full_log_toget_bullet_pts: The path to the log file for storing the extracted features to get bullet_points.
            - full_log_toget_summary: The path to the log file for storing the combined log messages to get summary.

        Returns:
            log_message: The combined log message containing the extracted features.
        """
        print("PRINTING INSIDE VIDEO PROCESSING")

        last_logged_time = datetime.now()
        print(last_logged_time)

        # Assuming process_* functions are coroutine functions or wrapped to be compatible with async
        task_1 = process_blink(frame, self.detector, self.lm_model,
                               self.L_start, self.L_end, self.R_start, self.R_end)
        task_2 = process_gaze(frame, self.detector, self.lm_model)
        task_3 = process_mouth(frame, self.lm_model, self.detector)
        task_4 = process_emotion(
            frame, self.face_classifier, self.classifier, self.detector)
        task_5 = headpose_process(frame, self.detector, self.lm_model)
        task_6 = process_frame_with_yolo(frame, self.model)
        task_7 = body_posture(
            frame, self.pose, self.mp_drawing, self.mp_pose, self.previous_angles)
        task_8 = asyncio.create_task(mark_attendance(
            known_encoding=image_encoding, frame=frame
        ))
        results = await asyncio.gather(task_1, task_2, task_3, task_4, task_5, task_6, task_7, task_8)

        _, blink_statuses = results[0]
        _, gaze_directions = results[1]
        _, lip_statuses = results[2]
        _, emotion_statuses = results[3]
        _, headpose_directions = results[4]
        _, log_message = results[5]
        _, movement_message, current_angles = results[6]
        log_attendance = results[7]
        # combined_frame = frame.copy()
        # Combine the results for visualization
        for i in range(len(blink_statuses)):
            blink_status = blink_statuses[i]
            gaze_direction = gaze_directions[i]
            lip_status = lip_statuses[i]
            emotion_status = emotion_statuses[i]
            # headpose_direction = headpose_directions[i]

            events_to_log = []
            # always attendance
            if log_attendance:
                events_to_log.append(f"Attendance : {log_attendance}")
                events_to_log.append(f"Head Pose: {headpose_directions}")

                if movement_message:
                    events_to_log.append(f"Body Postures:{movement_message}")

                if current_angles is not None:
                    self.previous_angles = current_angles

                # Always append detected objects into list
                events_to_log.append(f"Object Detected : {log_message}")

                # Always add emotion status to log list
                events_to_log.append(f"Emotion: {emotion_status}")

                # if headpose_direction in ["Left", "Right",'Center']:
                #     events_to_log.append(f"Head Pose: {headpose_direction}")
                if blink_status == "Blinking":
                    events_to_log.append(f"Blink Status: {blink_status}")
                if gaze_direction in ["Left", "Right"]:
                    events_to_log.append(f"Gaze Direction: {gaze_direction}")
                if lip_status in ["Closed", "Open"]:
                    events_to_log.append(f"Mouth Status: {lip_status}")
            else:
                events_to_log.append(f"Attendance : {log_attendance}")
            current_time = datetime.now()
            print(current_time)
            # if events_to_log and (current_time - last_logged_time).seconds >= 1:
            log_message = ", ".join(events_to_log)
            print(log_message)
            if not os.path.exists(r"modules/voice_bot/log_files"):
                os.makedirs(r"modules/voice_bot/log_files", exist_ok=True)
            log_event(
                log_message,  r"modules/voice_bot/log_files/full_logs_toget_summary.txt")
            log_event(
                log_message, r"modules/voice_bot/log_files/full_logs_toget_bullet_pts.txt")
            last_logged_time = current_time
            return log_message

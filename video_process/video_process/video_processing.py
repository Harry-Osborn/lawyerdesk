

import asyncio
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
from dotenv import load_dotenv
import keyboard
from modules.voice_bot.video_process.detect_object import process_frame_with_yolo
from concurrent.futures import ThreadPoolExecutor
from modules.voice_bot.video_process.body_postures import body_posture
import mediapipe as mp
from ultralytics import YOLO
from modules.voice_bot.video_process.detect_face import mark_attendance, known_image_encoding
load_dotenv()


async def video_processing(image, full_log_toget_bullet_pts, full_log_toget_summary,  frame_rate=30,):
    """
    Process video frames captured from a camera feed using computer vision techniques.

    Args:
        - cam (int): The camera feed to capture video frames from.
        - full_log_toget_bullet_pts (str): The file path to log the detected events to get bullet point.
        - full_log_toget_summary (str): The file path to log the detected events to get summary .
        - frame_rate (int, optional): The desired frame rate for video processing. Defaults to 30.

    Code Analysis:
    - Inputs:
        - `cam`: The camera feed to capture video frames from.
        - `full_log_toget_bullet_pts`: The file path to log the detected events in bullet point format.
        - `full_log_toget_summary`: The file path to log the detected events in summary format.
        - `frame_rate`: The desired frame rate for video processing.

    - Flow:
        1. Initialize the video capture from the specified camera feed.
        2. Load the necessary models and classifiers for face detection, facial landmark detection, emotion classification, and object detection.
        3. Create a thread pool executor to parallelize the processing of different aspects of human behavior.
        4. Continuously read video frames from the camera feed.
        5. Submit tasks to the thread pool executor for processing eye blinking, gaze direction, mouth movement, emotion, head pose, object detection, and body posture.
        6. Combine the results of the different processing tasks into a single frame for visualization.
        7. Log the detected events at regular intervals.
        8. Display the processed frames in real-time.
        9. Terminate the video processing when the 'q' key is pressed or when Ctrl+C is detected.

    """
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, frame_rate)
    detector = dlib.get_frontal_face_detector()
    lm_model = dlib.shape_predictor(
        r'loggers/saved_models/shape_predictor_68_face_landmarks.dat')
    L_start, L_end = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
    R_start, R_end = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    face_classifier = cv2.CascadeClassifier(
        r'loggers/saved_models/haarcascade_frontalface_default.xml')
    classifier = load_model(r'loggers/saved_models/Emotion_Detection.h5')
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose()
    # known_encoding = known_image_encoding(image_path = known_image)
    # Load the YOLOv8 model
    model = YOLO(r'loggers/saved_models/yolov8n.pt')
    encoding_image = await known_image_encoding(image)
    last_logged_time = datetime.now()
    previous_angles = None
    with ThreadPoolExecutor(max_workers=30) as executor:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Async processing tasks
            task_1 = asyncio.create_task(process_blink(
                frame.copy(), detector, lm_model, L_start, L_end, R_start, R_end))
            task_2 = asyncio.create_task(
                process_gaze(frame.copy(), detector, lm_model))
            task_3 = asyncio.create_task(
                process_mouth(frame.copy(), lm_model, detector))
            task_4 = asyncio.create_task(process_emotion(
                frame.copy(), face_classifier, classifier, detector))
            task_5 = asyncio.create_task(
                headpose_process(frame.copy(), detector, lm_model))
            task_6 = asyncio.create_task(
                process_frame_with_yolo(frame.copy(), model))
            task_7 = asyncio.create_task(body_posture(
                frame.copy(), pose, mp_drawing, mp_pose, previous_angles))
            task_8 = asyncio.create_task(mark_attendance(
                known_encoding=encoding_image, frame=frame.copy()
            ))

            # Gather all tasks
            results = await asyncio.gather(task_1, task_2, task_3, task_4, task_5, task_6, task_7, task_8)

            # Extract results
            _, blink_statuses = results[0]
            _, gaze_directions = results[1]
            _, lip_statuses = results[2]
            _, emotion_statuses = results[3]
            _, headpose_directions = results[4]
            _, log_message = results[5]
            frame_7, movement_message, current_angles = results[6]
            log_attendance = results[7]

            combined_frame = frame_7.copy()

            # Combine the results for visualization
            events_to_log = []

            for i in range(len(blink_statuses)):
                blink_status = blink_statuses[i]
                gaze_direction = gaze_directions[i]
                lip_status = lip_statuses[i]
                emotion_status = emotion_statuses[i]

                if log_attendance == 'Present':
                    events_to_log.append(f"Attendance : {log_attendance}")
                    events_to_log.append(f'Headpose : {headpose_directions}')
                    if movement_message:
                        events_to_log.append(
                            f"Body Postures:{movement_message}")

                    if current_angles is not None:
                        previous_angles = current_angles

                    # always attendance

                    # Always append detected objects into list
                    events_to_log.append(f"Object Detected : {log_message}")

                    # Always add emotion status to log list
                    events_to_log.append(f"Emotion: {emotion_status}")

                    if blink_status == "Blinking":
                        events_to_log.append(f"Blink Status: {blink_status}")
                    if gaze_direction in ["Left", "Right"]:
                        events_to_log.append(
                            f"Gaze Direction: {gaze_direction}")
                    if lip_status in ["Closed", "Open"]:
                        events_to_log.append(f"Mouth Status: {lip_status}")

                    else:  # Add this else condition to handle the case where no one is detected
                        events_to_log.append('No one detected')
                else:
                    events_to_log.append(log_attendance)
            current_time = datetime.now()

            if events_to_log:
                # log_event(', '.join(events_to_log), filename)
                log_event(', '.join(events_to_log), full_log_toget_bullet_pts)
                log_event(', '.join(events_to_log), full_log_toget_summary)
                last_logged_time = current_time

            # cv2.imshow('Combined Frame', combined_frame)
            cv2.imshow('YOLO Frame', combined_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if keyboard.is_pressed('ctrl+c'):
                print("Exiting due to Ctrl+C...")
                break
        cap.release()
        cv2.destroyAllWindows()


# cam = 0,
# image = r"images\front_side1.png"
# full_log_toget_bullet_pts = r'log_files\full_log_bp.txt'
# full_log_toget_summary = r"log_files\full_log_summ.txt"

# asyncio.run(video_processing(full_log_toget_bullet_pts=full_log_toget_bullet_pts, image=image,
#             full_log_toget_summary=full_log_toget_summary, frame_rate=20))

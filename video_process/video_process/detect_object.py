

import asyncio
from datetime import datetime
import cv2
from ultralytics import YOLO
from loggers.timer_decorator import timer_decorator


async def process_frame_with_yolo(frame, model):
    """
    Perform object tracking on a frame using a YOLO model.

    Args:
        - frame (array): The input frame on which object tracking will be performed.
        - model (YOLO): The YOLO model used for object tracking.

    Returns:
        - tuple: A tuple containing the annotated frame and a log message.

    Example Usage:
        - frame = ...
        - model = YOLO(...)
        - annotated_frame, log_message = process_frame_with_yolo(frame, model)

    Summary:
        The `process_frame_with_yolo` function takes a frame and a YOLO model as inputs. It uses the YOLO model to perform object tracking on the frame and returns an annotated frame and a log message.

    Code Analysis:
        - The function uses the YOLO model to track objects in the input frame.
        - It extracts the names of the detected objects from the results.
        - It counts the occurrences of each detected object.
        - It formats the object counts into a log message.
        - It returns the annotated frame and the log message.

    Outputs:
        annotated_frame (array): The input frame with bounding boxes and labels drawn around the detected objects.
        log_message (str): A log message containing the counts of each detected object.
    """
    # print("Printing inside process_frame_with_yolo function")

    # Run YOLOv8 tracking on the frame
    results = model.track(frame, persist=False)

    # Debug: Print raw results from the model
    # print("Raw Model Results:", results)

    # Take the first result if results is a list
    result = results[0] if isinstance(results, list) else results

    # Extract detected object names
    detected_objects = [result.names[int(i)] for i in result.boxes.cls]

    # Debug: Print all detected objects
    print("All Detected Objects:", detected_objects)

    obj_counts = {name: detected_objects.count(
        name) for name in detected_objects}

    # Debug: Print object counts
    print("Object Counts:", obj_counts)

    # Format and log the information
    log_message = f"{', '.join([f'{v} {k}' for k, v in obj_counts.items()])}"

    # Visualize the results on the frame
    annotated_frame = results[0].plot()
    # print("Printing inside process_frame_with_yolo function---------", log_message)
    return annotated_frame, log_message


# # Initialize video capture
# cap = cv2.VideoCapture(0)  # 0 for default camera

# # Initialize last_log_time and log_message
# last_log_time = datetime.now()
# log_message = ""
# model = YOLO(r"model\yolo_model\yolov8n.pt")
# while True:
#     ret, frame = cap.read()  # Capture frame
#     if not ret:
#         print("Failed to grab frame")
#         break

#     annotated_frame, log_message = asyncio.run(process_frame_with_yolo(
#         frame, model))  # Process frame with YOLO

#     # Display the resulting frame
#     cv2.imshow("YOLOv8 Annotated Frame", annotated_frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
#         break

# # Release the capture and destroy all OpenCV windows
# cap.release()
# cv2.destroyAllWindows()

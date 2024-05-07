

import asyncio
import os
import cv2
import face_recognition
from loggers.timer_decorator import timer_decorator


async def known_image_encoding(image_path):
    """
    Returns the 128-dimension face encoding of the image.

    Args:
        - image_path (str or path-like object): The path to the image file.

    Returns:
        numpy.ndarray: A 128-dimensional numpy array representing the face encoding of the image.
    """
    # Ensure image_path is a string or path-like object
    if not isinstance(image_path, (str, bytes, os.PathLike)):
        raise ValueError(
            f"Invalid image_path type. Expected a string or path-like object, got {type(image_path)}")

    try:
        # Load image
        img = face_recognition.load_image_file(image_path)
        # Get 128-dimension face encoding
        img_enc = face_recognition.face_encodings(img)[0]
        return img_enc
    except Exception as e:
        # Handle exceptions, such as file not found or face not detected
        print(f"An error occurred in known_image_encoding function: {e}")
        return None

# print(asyncio.run(known_image_encoding(r"known_image\front_side.jpg")))


async def mark_attendance(known_encoding, frame):
    """
    Marks the attendance of a person as "Present" or "Absent" based on face recognition.

    Args:
        - known_encoding (numpy.ndarray): A array representing the known face encoding used for comparison.
        - frame (numpy.ndarray): An image frame in BGR format.

    Returns:
        str: "Present" if the person is identified as present, "Absent - Another person is detected or Not looking at the center" if another person is detected or the person is not looking at the center, "Absent - No face detected" if no face is detected in the frame.

    Raises:
        Exception: If an error occurs in the mark attendance function.

    Example:
        known_encoding = np.array([0.1, 0.2, 0.3, ...])  # Known face encoding
        frame = cv2.imread('image.jpg')  # Input frame

        result = mark_attendance(known_encoding, frame)
        print(result)  # Output: 'Present' or 'Absent'
    """
    try:
        person_absent = True

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(
            rgb_frame, face_locations)

        # If no faces are detected, mark as Absent
        if len(face_locations) != 0:
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(
                    [known_encoding], face_encoding)
                if True in matches:
                    cv2.rectangle(frame, (left, top),
                                  (right, bottom), (0, 255, 0), 2)
                    cv2.putText(frame, "Present", (left, bottom + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    person_absent = False  # Set the flag to False as the person is identified as present
                    return 'Present'
                else:
                    return 'Absent - Another person is detected or Not looking at the center'
        else:
            return "Absent - No face detected"

    except Exception as e:
        print("An error occurred in mark attendance function: ", e)

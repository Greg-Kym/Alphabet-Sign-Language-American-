import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

latest_results = None

HAND_CONNECTIONS = [
    (), (), (), (),  # Thumb
    (), (), (), (),  # Index
    (), (), (), (),  # Middle
    (), (), (), (),  # Ring
    (), (), (), (),  # Pinky
    (), (), (), (),
]


def results_update(results, mp_image: mp.Image, timestamp_ms: int):
    global latest_results
    latest_results = results


baseOptions = python.BaseOptions(
    model_asset_path='Model/hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    num_hands=1,
    base_options=baseOptions,
    running_mode=vision.RunningMode.LIVE_STREAM,
    result_callback=results_update
)

with vision.HandLandmarker.create_from_options(options) as detector:

    cap = cv2.VideoCapture(0)

    while cap.isOpened():

        succ, frame = cap.read()

        if not succ or frame is None:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        time_stamp = int(time.time() * 1000)
        detector.detect_async(mp_image, time_stamp)

        if latest_results and latest_results.hand_landmarks:

            for hand_landmark in latest_results.hand_landmarks:

                for landmarks in hand_landmark:

                    x = int(landmarks.x * frame.shape[1])
                    y = int(landmarks.y * frame.shape[0])

                    cv2.circle(frame, (x, y), 5, (0, 0, 0), -1)

        cv2.imshow('Press ESC to Exit', frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

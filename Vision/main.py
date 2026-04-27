import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import torch
from model_structure import Sign_Model

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = Sign_Model(input=63, hidden_layers=8,
                   output=28).to(device)
model.load_state_dict(torch.load('Model/asl_model.pth', map_location=device))
model.eval()

latest_results = None

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
    (1, 5), (5, 6), (6, 7), (7, 8),  # Index
    (5, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (9, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # Pinky
    (17, 0)
]

labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
          'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
          'Space', 'Nothing']


def results_update(results, mp_image: mp.Image, timestamp_ms: int):
    global latest_results
    latest_results = results


def extract_landmarks(hand_landmarks):
    temp_list = []
    for lm in hand_landmarks:
        temp_list.extend([lm.x, lm.y, lm.z])

    base_x, base_y, base_z = temp_list[0], temp_list[1], temp_list[2]

    normalized_list = []
    for i in range(0, len(temp_list), 3):
        normalized_list.append(temp_list[i] - base_x)
        normalized_list.append(temp_list[i+1] - base_y)
        normalized_list.append(temp_list[i+2] - base_z)

    max_val = max(map(abs, normalized_list))
    if max_val != 0:
        normalized_list = [n / max_val for n in normalized_list]

    return normalized_list


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

                for CONNECTION in HAND_CONNECTIONS:

                    start = hand_landmark[CONNECTION[0]]
                    end = hand_landmark[CONNECTION[1]]

                    cv2.line(frame, (int(start.x * frame.shape[1]), int(
                        start.y * frame.shape[0])), (int(end.x * frame.shape[1]), int(end.y * frame.shape[0])), (246, 89, 155), 2)

                input_data = extract_landmarks(hand_landmark)

                input_tensor = torch.tensor(
                    input_data, dtype=torch.float32).unsqueeze(0).to(device)

                with torch.inference_mode():
                    prediction = model(input_tensor)
                    predicted_index = torch.argmax(prediction, dim=1).item()

                predicted_letter = labels[predicted_index]

                print(f"Predicted Sign: {predicted_letter}")

                cv2.putText(frame, f'Sign: {predicted_letter}', (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('Press ESC to Exit', frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

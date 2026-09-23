import cv2
import mediapipe as mp
import pickle
import numpy as np
import os
import time
import subprocess
from collections import deque

# --- Load the trained model ---
MODEL_FILE = "asl_model.pkl"
if not os.path.exists(MODEL_FILE):
    print("ERROR: asl_model.pkl not found. Run asl_phase3_train.py first.")
    exit()

with open(MODEL_FILE, "rb") as f:
    model = pickle.load(f)
print("Model loaded!")

# --- MediaPipe Setup ---
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = "hand_landmarker.task"
latest_result = None

def result_callback(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH,
        delegate=BaseOptions.Delegate.CPU
    ),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7,
    result_callback=result_callback
)

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

def draw_landmarks(frame, hand_landmarks_list):
    h, w, _ = frame.shape
    for hand_landmarks in hand_landmarks_list:
        points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
        for start, end in HAND_CONNECTIONS:
            cv2.line(frame, points[start], points[end], (0, 200, 255), 2)
        for pt in points:
            cv2.circle(frame, pt, 5, (255, 255, 255), -1)
            cv2.circle(frame, pt, 5, (0, 150, 255), 1)

def get_landmark_coords(hand_landmarks):
    coords = []
    for lm in hand_landmarks:
        coords.extend([lm.x, lm.y, lm.z])
    return coords

# --- Translation State ---
# Buffer of recent predictions — only commit a letter when it's stable
prediction_buffer = deque(maxlen=15)
current_sentence = ""
last_committed_letter = ""
last_commit_time = 0
COMMIT_DELAY = 1.2  # seconds to hold a sign before it's added
NO_HAND_CLEAR_DELAY = 2.0  # seconds of no hand before adding a space

last_hand_time = time.time()
commit_progress = 0.0

print("ASL Live Translator running!")
print("  Hold a sign steady to add a letter")
print("  Lower your hand for 2 seconds to add a space")
print("  Press S to speak the current sentence")
print("  Press BACKSPACE to delete last character")
print("  Press C to clear the sentence")
print("  Press Q to quit\n")

cap = cv2.VideoCapture(0)
timestamp_ms = 0

with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        landmarker.detect_async(mp_image, timestamp_ms)
        timestamp_ms += 33

        hand_detected = latest_result and latest_result.hand_landmarks
        predicted_letter = ""
        confidence = 0.0

        if hand_detected:
            last_hand_time = time.time()
            draw_landmarks(frame, latest_result.hand_landmarks)

            # Get prediction
            coords = get_landmark_coords(latest_result.hand_landmarks[0])
            coords_array = np.array(coords).reshape(1, -1)
            proba = model.predict_proba(coords_array)[0]
            confidence = np.max(proba)
            predicted_letter = model.classes_[np.argmax(proba)]

            # Add to buffer if confidence is high enough
            if confidence > 0.6:
                prediction_buffer.append(predicted_letter)
            else:
                prediction_buffer.clear()

            # Check if buffer is stable (same letter dominates)
            if len(prediction_buffer) == prediction_buffer.maxlen:
                most_common = max(set(prediction_buffer), key=prediction_buffer.count)
                count = prediction_buffer.count(most_common)
                stability = count / len(prediction_buffer)

                if stability > 0.8:
                    now = time.time()
                    if most_common != last_committed_letter:
                        last_committed_letter = most_common
                        last_commit_time = now
                        commit_progress = 0.0
                    else:
                        elapsed = now - last_commit_time
                        commit_progress = min(elapsed / COMMIT_DELAY, 1.0)
                        if elapsed >= COMMIT_DELAY:
                            current_sentence += most_common
                            print(f"Added: {most_common} → '{current_sentence}'")
                            last_committed_letter = ""
                            last_commit_time = 0
                            commit_progress = 0.0
                            prediction_buffer.clear()
                else:
                    commit_progress = 0.0
        else:
            # No hand — maybe add a space
            prediction_buffer.clear()
            commit_progress = 0.0
            elapsed_no_hand = time.time() - last_hand_time
            if elapsed_no_hand > NO_HAND_CLEAR_DELAY and current_sentence and not current_sentence.endswith(" "):
                current_sentence += " "
                last_committed_letter = ""
                print(f"Added space → '{current_sentence}'")

        # --- Draw UI ---

        # Top bar
        cv2.rectangle(frame, (0, 0), (w, 120), (25, 25, 25), -1)

        # Predicted letter (big)
        if predicted_letter and hand_detected:
            color = (0, 255, 180) if confidence > 0.6 else (0, 150, 255)
            cv2.putText(frame, predicted_letter, (20, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 3.5, color, 5)
            conf_text = f"{confidence*100:.0f}%"
            cv2.putText(frame, conf_text, (110, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (180, 180, 180), 2)

        # Commit progress ring/bar
        if commit_progress > 0:
            bar_w = int(commit_progress * (w - 40))
            cv2.rectangle(frame, (20, 105), (w - 20, 115), (60, 60, 60), -1)
            cv2.rectangle(frame, (20, 105), (20 + bar_w, 115), (0, 255, 180), -1)

        # Status
        status = "Hand detected" if hand_detected else "Waiting for hand..."
        status_color = (0, 255, 100) if hand_detected else (100, 100, 100)
        cv2.putText(frame, status, (w - 260, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)

        # Sentence output box at bottom
        cv2.rectangle(frame, (0, h - 80), (w, h), (20, 20, 20), -1)
        cv2.putText(frame, "Translation:", (15, h - 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 120, 120), 1)

        # Truncate sentence display if too long
        display_sentence = current_sentence[-35:] if len(current_sentence) > 35 else current_sentence
        cv2.putText(frame, display_sentence + "|", (15, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)

        cv2.imshow("ASL Live Translator", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            if current_sentence.strip():
                print(f"Speaking: '{current_sentence.strip()}'")
                subprocess.Popen(["say", current_sentence.strip()])
        elif key == ord('c'):
            current_sentence = ""
            print("Sentence cleared.")
        elif key == 8 or key == 127:  # Backspace
            current_sentence = current_sentence[:-1]
            print(f"Deleted → '{current_sentence}'")

cap.release()
cv2.destroyAllWindows()
print(f"\nFinal translation: '{current_sentence}'")

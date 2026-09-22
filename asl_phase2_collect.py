import cv2
import mediapipe as mp
import csv
import os
import time

# --- MediaPipe Setup ---
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = "hand_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print("ERROR: hand_landmarker.task not found. Run asl_phase1.py first to download it.")
    exit()

latest_result = None

def result_callback(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
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
    """Flatten 21 landmarks into a list of 63 values (x, y, z per landmark)"""
    coords = []
    for lm in hand_landmarks:
        coords.extend([lm.x, lm.y, lm.z])
    return coords

# --- Data Collection Setup ---
LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
SAMPLES_PER_LETTER = 30   # How many samples to collect per letter
CSV_FILE = "asl_training_data.csv"

# Load existing data so we can resume if interrupted
existing_labels = set()
if os.path.exists(CSV_FILE):
    with open(CSV_FILE, "r") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if row:
                existing_labels.add(row[0])
    print(f"Resuming — already collected: {sorted(existing_labels)}")

# Filter to only letters we still need
letters_needed = [l for l in LETTERS if l not in existing_labels]

if not letters_needed:
    print("All letters already collected! Check asl_training_data.csv")
    exit()

# Write CSV header if file doesn't exist
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["label"] + [f"x{i}" if j==0 else f"y{i}" if j==1 else f"z{i}"
                               for i in range(21) for j in range(3)]
        writer.writerow(header)

print(f"\nLetters to collect: {letters_needed}")
print("Instructions:")
print("  - Hold up the ASL sign shown on screen")
print("  - Press SPACE to record a sample (do this 30 times per letter)")
print("  - Press N to skip to the next letter")
print("  - Press Q to quit and save progress\n")

# --- Main Loop ---
cap = cv2.VideoCapture(0)
timestamp_ms = 0
letter_idx = 0
sample_count = 0
feedback_msg = ""
feedback_timer = 0

with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened() and letter_idx < len(letters_needed):
        current_letter = letters_needed[letter_idx]
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        landmarker.detect_async(mp_image, timestamp_ms)
        timestamp_ms += 33

        hand_detected = latest_result and latest_result.hand_landmarks

        if hand_detected:
            draw_landmarks(frame, latest_result.hand_landmarks)

        # --- UI Overlay ---
        h, w, _ = frame.shape

        # Dark banner at top
        cv2.rectangle(frame, (0, 0), (w, 110), (30, 30, 30), -1)

        # Current letter (big)
        cv2.putText(frame, current_letter, (30, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 180), 4)

        # Progress
        progress_text = f"Letter {letter_idx+1}/{len(letters_needed)}   Samples: {sample_count}/{SAMPLES_PER_LETTER}"
        cv2.putText(frame, progress_text, (130, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        # Instructions
        cv2.putText(frame, "SPACE = record sample   N = skip   Q = quit", (130, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1)

        # Hand status
        status_color = (0, 255, 0) if hand_detected else (0, 0, 255)
        status_text = "Hand detected" if hand_detected else "No hand detected"
        cv2.putText(frame, status_text, (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        # Progress bar
        bar_width = int((sample_count / SAMPLES_PER_LETTER) * (w - 40))
        cv2.rectangle(frame, (20, h - 55), (w - 20, h - 40), (60, 60, 60), -1)
        cv2.rectangle(frame, (20, h - 55), (20 + bar_width, h - 40), (0, 255, 180), -1)

        # Feedback flash
        if feedback_msg and time.time() < feedback_timer:
            cv2.putText(frame, feedback_msg, (w//2 - 120, h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 100), 3)

        cv2.imshow("ASL Data Collector - Phase 2", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("Quitting and saving progress.")
            break

        elif key == ord('n'):
            print(f"Skipping {current_letter} (collected {sample_count} samples)")
            letter_idx += 1
            sample_count = 0

        elif key == ord(' '):
            if not hand_detected:
                feedback_msg = "No hand! Show your hand first."
                feedback_timer = time.time() + 1.5
            else:
                # Save the landmark data
                coords = get_landmark_coords(latest_result.hand_landmarks[0])
                with open(CSV_FILE, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([current_letter] + coords)

                sample_count += 1
                feedback_msg = f"Saved! ({sample_count}/{SAMPLES_PER_LETTER})"
                feedback_timer = time.time() + 0.5

                if sample_count >= SAMPLES_PER_LETTER:
                    print(f"✓ Finished collecting '{current_letter}'")
                    feedback_msg = f"'{current_letter}' done! Next letter..."
                    feedback_timer = time.time() + 1.5
                    cv2.waitKey(1500)
                    letter_idx += 1
                    sample_count = 0

cap.release()
cv2.destroyAllWindows()

# Summary
print(f"\nDone! Training data saved to: {CSV_FILE}")
if os.path.exists(CSV_FILE):
    with open(CSV_FILE, "r") as f:
        rows = sum(1 for _ in f) - 1
    print(f"Total samples collected: {rows}")

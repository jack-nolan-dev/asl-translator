# ASL Sign Language Translator

Real-time American Sign Language recognition using computer vision and machine learning. Point your webcam at hand gestures and the system identifies ASL letters in real time.

## How It Works

Live webcam feed → MediaPipe hand landmark detection → Random Forest classifier → predicted ASL letter displayed on screen.

**Accuracy:** ~79% on test data

## Architecture

The project is built in 4 phases, each a standalone script:

| Phase | Script | What it does |
|-------|--------|-------------|
| 1 | `asl_phase1.py` | Real-time hand tracking with MediaPipe + OpenCV |
| 2 | `asl_phase2_collect.py` | Collect training data — capture hand landmarks for each ASL letter |
| 3 | `asl_phase3_train.py` | Train a Random Forest classifier on collected gesture data |
| 4 | `asl_phase4_live.py` | Live inference — webcam feed with real-time ASL letter prediction |

## Tech Stack

- **Python**
- **MediaPipe** — hand landmark detection (21 key points per hand)
- **OpenCV** — webcam capture and display
- **scikit-learn** — Random Forest classifier
- **NumPy / pandas** — data processing

## Run It

```bash
pip install mediapipe opencv-python scikit-learn numpy pandas

# Phase 1: Test hand tracking
python asl_phase1.py

# Phase 4: Run live translator (requires trained model)
python asl_phase4_live.py
```

## Files

- `asl_training_data.csv` — collected hand landmark training data
- `asl_model.pkl` — trained Random Forest model
- `confusion_matrix.png` — model evaluation results

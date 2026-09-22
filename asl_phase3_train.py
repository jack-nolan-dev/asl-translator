import csv
import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load the training data ---
CSV_FILE = "asl_training_data.csv"

if not os.path.exists(CSV_FILE):
    print("ERROR: asl_training_data.csv not found. Run phase 2 first.")
    exit()

print("Loading training data...")
X = []  # Features (landmark coordinates)
y = []  # Labels (letters)

with open(CSV_FILE, "r") as f:
    reader = csv.reader(f)
    next(reader)  # Skip header
    for row in reader:
        if row:
            y.append(row[0])           # First column = label
            X.append([float(v) for v in row[1:]])  # Rest = coordinates

X = np.array(X)
y = np.array(y)

print(f"Loaded {len(X)} samples across {len(set(y))} letters: {sorted(set(y))}")

# --- Split into training and test sets ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")

# --- Train the classifier ---
print("\nTraining Random Forest classifier...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1  # Use all CPU cores
)
model.fit(X_train, y_train)
print("Training complete!")

# --- Evaluate ---
y_pred = model.predict(X_test)
accuracy = (y_pred == y_test).mean() * 100
print(f"\nTest Accuracy: {accuracy:.1f}%")

print("\nPer-letter breakdown:")
print(classification_report(y_test, y_pred))

# --- Save the model ---
MODEL_FILE = "asl_model.pkl"
with open(MODEL_FILE, "wb") as f:
    pickle.dump(model, f)
print(f"Model saved to: {MODEL_FILE}")

# --- Confusion matrix ---
print("\nGenerating confusion matrix...")
labels = sorted(set(y))
cm = confusion_matrix(y_test, y_pred, labels=labels)

plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels)
plt.title("ASL Classifier — Confusion Matrix", fontsize=16)
plt.xlabel("Predicted Letter", fontsize=12)
plt.ylabel("Actual Letter", fontsize=12)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()
print("Confusion matrix saved to: confusion_matrix.png")

print("\nAll done! Run asl_phase4_live.py to try it live.")

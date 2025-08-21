# evaluate_model.py
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model
from data_loader import load_data

# === CONFIG ===
BASE_DIR = r"C:\users\harsh\Downloads\TrainingData\dataset_split"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
MODEL_PATH = "best_model_val_acc.h5"

# === LOAD DATA AND MODEL ===
_, _, test_loader = load_data(BASE_DIR, IMG_SIZE, BATCH_SIZE)
model = load_model(MODEL_PATH)

# === PREDICT ===
y_true = test_loader.classes
class_labels = list(test_loader.class_indices.keys())
y_pred = np.argmax(model.predict(test_loader), axis=1)

# === CONFUSION MATRIX ===
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
#plt.savefig("confusion_matrix.png")
plt.show()

# === CLASSIFICATION REPORT ===
print("Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_labels))

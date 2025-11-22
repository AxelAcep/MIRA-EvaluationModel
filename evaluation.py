"""
Model Evaluation Script
Evaluasi Face Recognition dan YOLO Human Detection
"""

import cv2
import numpy as np
import face_recognition
import pickle
import os
from pathlib import Path
import time
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    precision_recall_fscore_support,
    roc_curve, 
    auc
)
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
from collections import defaultdict

print("="*70)
print(" MODEL EVALUATION SCRIPT ".center(70, "="))
print("="*70)

# ==================== CONFIGURATION ====================
DATASET_PATH = "Dataset"  # Folder berisi subfolder per NIM/nama
MODEL_PATH = "model.dat"  # Face encoding model
YOLO_PATH = "yolo11n.pt"  # YOLO model
OUTPUT_DIR = "evaluation_results"  # Folder output hasil evaluasi

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== LOAD MODELS ====================
print("\n[1/5] Loading Models...")

# Load Face Recognition Model
try:
    with open(MODEL_PATH, 'rb') as f:
        face_data = pickle.load(f)
        known_face_encodings = face_data['encodings']
        known_face_names = face_data['names']
    print(f"✓ Face Recognition Model loaded: {len(known_face_names)} identities")
    print(f"  Identities: {set(known_face_names)}")
except Exception as e:
    print(f"✗ Error loading face model: {e}")
    exit(1)

# Load YOLO Model
try:
    yolo_model = YOLO(YOLO_PATH)
    print(f"✓ YOLO Model loaded: {YOLO_PATH}")
except Exception as e:
    print(f"✗ Error loading YOLO model: {e}")
    exit(1)

# ==================== LOAD TEST DATASET ====================
print(f"\n[2/5] Loading Test Dataset from: {DATASET_PATH}")

test_images = []
test_labels = []

if not os.path.exists(DATASET_PATH):
    print(f"✗ Dataset folder not found: {DATASET_PATH}")
    exit(1)

# Scan dataset folders
for person_folder in sorted(os.listdir(DATASET_PATH)):
    person_path = os.path.join(DATASET_PATH, person_folder)
    
    if not os.path.isdir(person_path):
        continue
    
    image_files = [f for f in os.listdir(person_path) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"  - {person_folder}: {len(image_files)} images")
    
    for img_file in image_files:
        img_path = os.path.join(person_path, img_file)
        test_images.append(img_path)
        test_labels.append(person_folder)

print(f"\n✓ Total test images loaded: {len(test_images)}")
print(f"✓ Unique identities in test set: {len(set(test_labels))}")

if len(test_images) == 0:
    print("✗ No test images found!")
    exit(1)

# ==================== FACE RECOGNITION EVALUATION ====================
print("\n[3/5] Evaluating Face Recognition Model...")

predictions = []
true_labels = []
confidences = []
latencies = []
failed_detections = 0

for idx, (img_path, true_label) in enumerate(zip(test_images, test_labels)):
    if (idx + 1) % 10 == 0:
        print(f"  Processing: {idx + 1}/{len(test_images)}", end='\r')
    
    # Load image
    image = cv2.imread(img_path)
    if image is None:
        print(f"\n  Warning: Could not load {img_path}")
        continue
    
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Measure latency
    start_time = time.time()
    
    # Detect faces
    face_locations = face_recognition.face_locations(rgb_image)
    
    if len(face_locations) == 0:
        failed_detections += 1
        predictions.append("Unknown")
        true_labels.append(true_label)
        confidences.append(0.0)
        latencies.append((time.time() - start_time) * 1000)
        continue
    
    # Get face encoding
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
    
    if len(face_encodings) == 0:
        failed_detections += 1
        predictions.append("Unknown")
        true_labels.append(true_label)
        confidences.append(0.0)
        latencies.append((time.time() - start_time) * 1000)
        continue
    
    # Compare with known faces
    face_encoding = face_encodings[0]
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    
    latency = (time.time() - start_time) * 1000
    latencies.append(latency)
    
    # Find best match
    if len(face_distances) > 0:
        best_match_index = np.argmin(face_distances)
        confidence = (1 - face_distances[best_match_index]) * 100
        
        if matches[best_match_index] and confidence >= 60:
            predicted_name = known_face_names[best_match_index]
        else:
            predicted_name = "Unknown"
            confidence = 0.0
    else:
        predicted_name = "Unknown"
        confidence = 0.0
    
    predictions.append(predicted_name)
    true_labels.append(true_label)
    confidences.append(confidence)

print(f"\n✓ Face Recognition Evaluation Complete!")
print(f"  - Total predictions: {len(predictions)}")
print(f"  - Failed detections: {failed_detections}")
print(f"  - Average latency: {np.mean(latencies):.2f} ms")

# ==================== GENERATE FACE RECOGNITION METRICS ====================
print("\n[4/5] Generating Face Recognition Metrics...")

# Create comprehensive figure
fig = plt.figure(figsize=(20, 12))
fig.suptitle('Face Recognition Model Evaluation', fontsize=16, fontweight='bold')

# 1. Confusion Matrix
ax1 = plt.subplot(2, 4, 1)
unique_labels = sorted(list(set(true_labels + predictions)))
cm = confusion_matrix(true_labels, predictions, labels=unique_labels)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=unique_labels, yticklabels=unique_labels, 
            ax=ax1, cbar_kws={'label': 'Count'})
ax1.set_title('Confusion Matrix', fontweight='bold')
ax1.set_ylabel('True Label')
ax1.set_xlabel('Predicted Label')
plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
plt.setp(ax1.get_yticklabels(), rotation=0)

# 2. ROC Curve
ax2 = plt.subplot(2, 4, 2)
try:
    # Binary: Correct vs Incorrect
    y_true_binary = [1 if t == p else 0 for t, p in zip(true_labels, predictions)]
    confidence_scores = np.array(confidences) / 100.0
    
    if len(set(y_true_binary)) > 1:
        fpr, tpr, thresholds = roc_curve(y_true_binary, confidence_scores)
        roc_auc = auc(fpr, tpr)
        ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
        ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC Curve', fontweight='bold')
        ax2.legend(loc="lower right")
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
except Exception as e:
    ax2.text(0.5, 0.5, f'Error: {str(e)[:30]}', ha='center', va='center')
    ax2.set_title('ROC Curve', fontweight='bold')

# 3. Precision, Recall, F1 per Individual
ax3 = plt.subplot(2, 4, 3)
precision, recall, f1, support = precision_recall_fscore_support(
    true_labels, predictions, labels=unique_labels, zero_division=0
)
x = np.arange(len(unique_labels))
width = 0.25
ax3.bar(x - width, precision, width, label='Precision', color='skyblue', edgecolor='black')
ax3.bar(x, recall, width, label='Recall', color='lightgreen', edgecolor='black')
ax3.bar(x + width, f1, width, label='F1-Score', color='salmon', edgecolor='black')
ax3.set_xlabel('Individual')
ax3.set_ylabel('Score')
ax3.set_title('Precision, Recall, F1-Score', fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(unique_labels, rotation=45, ha='right')
ax3.legend()
ax3.grid(axis='y', alpha=0.3)
ax3.set_ylim([0, 1.1])

# 4. Support (Sample Count) per Individual
ax4 = plt.subplot(2, 4, 4)
ax4.bar(unique_labels, support, color='purple', alpha=0.7, edgecolor='black')
ax4.set_xlabel('Individual')
ax4.set_ylabel('Number of Samples')
ax4.set_title('Test Samples per Individual', fontweight='bold')
plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
ax4.grid(axis='y', alpha=0.3)
for i, v in enumerate(support):
    ax4.text(i, v + 0.5, str(v), ha='center', va='bottom', fontweight='bold')

# 5. Latency Distribution
ax5 = plt.subplot(2, 4, 5)
ax5.hist(latencies, bins=30, color='teal', alpha=0.7, edgecolor='black')
ax5.axvline(np.mean(latencies), color='red', linestyle='--', linewidth=2,
           label=f'Mean: {np.mean(latencies):.2f} ms')
ax5.axvline(np.median(latencies), color='orange', linestyle='--', linewidth=2,
           label=f'Median: {np.median(latencies):.2f} ms')
ax5.set_xlabel('Latency (ms/face)')
ax5.set_ylabel('Frequency')
ax5.set_title('Face Recognition Latency', fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. Latency Box Plot
ax6 = plt.subplot(2, 4, 6)
bp = ax6.boxplot(latencies, vert=True, patch_artist=True)
bp['boxes'][0].set_facecolor('lightblue')
ax6.set_ylabel('Latency (ms)')
ax6.set_title('Latency Distribution', fontweight='bold')
ax6.grid(axis='y', alpha=0.3)
ax6.set_xticklabels(['All Predictions'])

# 7. Confidence Distribution
ax7 = plt.subplot(2, 4, 7)
filtered_conf = [c for c in confidences if c > 0]
if len(filtered_conf) > 0:
    ax7.hist(filtered_conf, bins=20, color='orange', alpha=0.7, edgecolor='black')
    ax7.axvline(np.mean(filtered_conf), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(filtered_conf):.1f}%')
    ax7.set_xlabel('Confidence (%)')
    ax7.set_ylabel('Frequency')
    ax7.set_title('Confidence Distribution', fontweight='bold')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
else:
    ax7.text(0.5, 0.5, 'No confident\npredictions', ha='center', va='center')
    ax7.set_title('Confidence Distribution', fontweight='bold')

# 8. Statistics Summary
ax8 = plt.subplot(2, 4, 8)
ax8.axis('off')

# Calculate accuracy
accuracy = sum([1 for t, p in zip(true_labels, predictions) if t == p]) / len(true_labels) * 100

stats_text = f"""
╔══════════════════════════════════╗
║   FACE RECOGNITION STATISTICS    ║
╚══════════════════════════════════╝

Total Samples: {len(true_labels)}
Failed Detections: {failed_detections}

Overall Accuracy: {accuracy:.2f}%

Latency Statistics:
  • Mean:    {np.mean(latencies):.2f} ms
  • Median:  {np.median(latencies):.2f} ms
  • Min:     {np.min(latencies):.2f} ms
  • Max:     {np.max(latencies):.2f} ms
  • Std:     {np.std(latencies):.2f} ms

Confidence Statistics:
  • Mean:    {np.mean(filtered_conf) if filtered_conf else 0:.2f}%
  • Median:  {np.median(filtered_conf) if filtered_conf else 0:.2f}%

Best Performing Identity:
  {unique_labels[np.argmax(f1)]}: F1={np.max(f1):.3f}
"""

ax8.text(0.1, 0.5, stats_text, transform=ax8.transAxes,
        fontsize=10, verticalalignment='center', family='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/face_recognition_evaluation.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {OUTPUT_DIR}/face_recognition_evaluation.png")

# Print detailed classification report
print("\n" + "="*70)
print("CLASSIFICATION REPORT")
print("="*70)
print(classification_report(true_labels, predictions, zero_division=0))

# ==================== YOLO HUMAN DETECTION EVALUATION ====================
print("\n[5/5] Evaluating YOLO Human Detection...")

yolo_detections = []
yolo_confidences = []
yolo_latencies = []
images_with_detection = 0
images_without_detection = 0

for idx, img_path in enumerate(test_images):
    if (idx + 1) % 10 == 0:
        print(f"  Processing: {idx + 1}/{len(test_images)}", end='\r')
    
    image = cv2.imread(img_path)
    if image is None:
        continue
    
    # Measure latency
    start_time = time.time()
    results = yolo_model(image, verbose=False)
    latency = (time.time() - start_time) * 1000
    yolo_latencies.append(latency)
    
    # Check for person detections
    person_detected = False
    for result in results[0].boxes:
        conf = result.conf[0].item()
        cls = result.cls[0].item()
        
        if yolo_model.names[int(cls)] == "person" and conf > 0.3:
            person_detected = True
            yolo_confidences.append(conf * 100)
    
    if person_detected:
        images_with_detection += 1
    else:
        images_without_detection += 1

print(f"\n✓ YOLO Evaluation Complete!")
print(f"  - Images with person detection: {images_with_detection}")
print(f"  - Images without detection: {images_without_detection}")
print(f"  - Detection rate: {images_with_detection/len(test_images)*100:.1f}%")
print(f"  - Average latency: {np.mean(yolo_latencies):.2f} ms")

# Generate YOLO metrics
fig2 = plt.figure(figsize=(16, 10))
fig2.suptitle('YOLO Human Detection Evaluation', fontsize=16, fontweight='bold')

# 1. Detection Rate
ax1 = plt.subplot(2, 3, 1)
labels_det = ['Detected', 'Not Detected']
sizes = [images_with_detection, images_without_detection]
colors = ['lightgreen', 'lightcoral']
explode = (0.1, 0)
ax1.pie(sizes, explode=explode, labels=labels_det, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=90)
ax1.set_title('Detection Rate', fontweight='bold')

# 2. Confidence Distribution
ax2 = plt.subplot(2, 3, 2)
if len(yolo_confidences) > 0:
    ax2.hist(yolo_confidences, bins=30, color='green', alpha=0.7, edgecolor='black')
    ax2.axvline(np.mean(yolo_confidences), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(yolo_confidences):.1f}%')
    ax2.set_xlabel('Confidence (%)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Detection Confidence Distribution', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

# 3. Confidence vs Threshold
ax3 = plt.subplot(2, 3, 3)
if len(yolo_confidences) > 0:
    thresholds = np.linspace(30, 95, 20)
    detection_counts = [sum(1 for c in yolo_confidences if c >= t) for t in thresholds]
    ax3.plot(thresholds, detection_counts, 'b-o', linewidth=2, markersize=6)
    ax3.set_xlabel('Confidence Threshold (%)')
    ax3.set_ylabel('Number of Detections')
    ax3.set_title('Detections vs Threshold', fontweight='bold')
    ax3.grid(True, alpha=0.3)

# 4. Latency Distribution
ax4 = plt.subplot(2, 3, 4)
ax4.hist(yolo_latencies, bins=30, color='purple', alpha=0.7, edgecolor='black')
ax4.axvline(np.mean(yolo_latencies), color='red', linestyle='--', linewidth=2,
           label=f'Mean: {np.mean(yolo_latencies):.2f} ms')
ax4.set_xlabel('Latency (ms/image)')
ax4.set_ylabel('Frequency')
ax4.set_title('YOLO Inference Latency', fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. Simulated Precision-Recall Curve
ax5 = plt.subplot(2, 3, 5)
if len(yolo_confidences) > 0:
    sorted_conf = sorted(yolo_confidences, reverse=True)
    precisions = []
    recalls = []
    
    for threshold in np.linspace(30, 95, 30):
        tp = sum(1 for c in sorted_conf if c >= threshold)
        fp = max(0, len(sorted_conf) - tp)
        fn = images_without_detection
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
    
    ax5.plot(recalls, precisions, 'r-', linewidth=2)
    ax5.fill_between(recalls, precisions, alpha=0.3, color='red')
    ax5.set_xlabel('Recall')
    ax5.set_ylabel('Precision')
    ax5.set_title('Precision-Recall Curve', fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim([0.0, 1.0])
    ax5.set_ylim([0.0, 1.05])
    
    # Calculate AP
    ap = np.trapz(precisions, recalls)
    ax5.text(0.05, 0.95, f'AP: {ap:.3f}', transform=ax5.transAxes,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            verticalalignment='top')

# 6. Statistics Summary
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

yolo_stats = f"""
╔══════════════════════════════════╗
║   YOLO DETECTION STATISTICS      ║
╚══════════════════════════════════╝

Total Images Tested: {len(test_images)}

Detection Results:
  • Detected:     {images_with_detection}
  • Not Detected: {images_without_detection}
  • Detection Rate: {images_with_detection/len(test_images)*100:.1f}%

Confidence Statistics:
  • Mean:   {np.mean(yolo_confidences) if yolo_confidences else 0:.2f}%
  • Median: {np.median(yolo_confidences) if yolo_confidences else 0:.2f}%
  • Min:    {np.min(yolo_confidences) if yolo_confidences else 0:.2f}%
  • Max:    {np.max(yolo_confidences) if yolo_confidences else 0:.2f}%

Latency Statistics:
  • Mean:   {np.mean(yolo_latencies):.2f} ms
  • Median: {np.median(yolo_latencies):.2f} ms
  • Min:    {np.min(yolo_latencies):.2f} ms
  • Max:    {np.max(yolo_latencies):.2f} ms

Detections > 50%: {sum(1 for c in yolo_confidences if c > 50)}
Detections > 70%: {sum(1 for c in yolo_confidences if c > 70)}
Detections > 90%: {sum(1 for c in yolo_confidences if c > 90)}
"""

ax6.text(0.1, 0.5, yolo_stats, transform=ax6.transAxes,
        fontsize=10, verticalalignment='center', family='monospace',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/yolo_detection_evaluation.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {OUTPUT_DIR}/yolo_detection_evaluation.png")

# ==================== FINAL SUMMARY ====================
print("\n" + "="*70)
print(" EVALUATION COMPLETE ".center(70, "="))
print("="*70)
print(f"\nResults saved to: {OUTPUT_DIR}/")
print("  - face_recognition_evaluation.png")
print("  - yolo_detection_evaluation.png")
print("\nAll metrics have been generated successfully! 🎉")
print("="*70)
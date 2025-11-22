import cv2
from ultralytics import YOLO
import time
import numpy as np
import face_recognition
import pickle
import keyboard

# --- NEW IMPORTS FOR MONITORING AND PLOTTING ---
import psutil
from pynvml import *
import matplotlib.pyplot as plt
from collections import deque
import os
import threading

# Initialize YOLO model with GPU
model = YOLO('yolo11n.pt').to('cuda')

# Camera setup
camera = cv2.VideoCapture(1)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
camera.set(cv2.CAP_PROP_FPS, 120)

# Load known faces
known_face_encodings = []
known_face_names = []

with open("model.dat", 'rb') as f:
    data = pickle.load(f)
    known_face_encodings = data['encodings']
    known_face_names = data['names']

# Tracking variables
id_tags = {}
active_person_tracks = {}
used_person_ids = set()
next_person_id = 1

# Presence tracking variables
person_presence_start_time = {}
person_total_presence_time = {}

# Timing controls
face_recognition_interval = 3
last_face_recog_time = 0

# Performance metrics
frame_count = 0
last_fps_time = time.time()
fps = 0

# --- Monitoring Data Storage (Now using standard lists to store ALL data) ---
MAX_LIVE_DATA_POINTS = 300

# Live plotting deques (for current window view)
cpu_usage_live = deque(maxlen=MAX_LIVE_DATA_POINTS)
ram_usage_live = deque(maxlen=MAX_LIVE_DATA_POINTS)
gpu_usage_live = deque(maxlen=MAX_LIVE_DATA_POINTS)
vram_usage_live = deque(maxlen=MAX_LIVE_DATA_POINTS)
yolo_confidence_live = deque(maxlen=MAX_LIVE_DATA_POINTS) # Perubahan nama variabel
fps_live = deque(maxlen=MAX_LIVE_DATA_POINTS)
time_live = deque(maxlen=MAX_LIVE_DATA_POINTS)

# Full history lists (for saving all data)
all_cpu_usage = []
all_ram_usage = []
all_gpu_usage = []
all_vram_usage = []
all_yolo_confidence = [] # Perubahan nama variabel
all_fps = []
all_time = []

last_monitor_update_time = time.time()
monitor_update_interval = 1

# --- GPU VRAM Capacity ---
VRAM_CAPACITY_MB = 4000

# --- Initialize NVML for GPU monitoring ---
try:
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    gpu_monitoring_enabled = True
    print("GPU monitoring initialized.")
except NVMLError as error:
    print(f"Failed to initialize NVML: {error}. GPU monitoring disabled.")
    gpu_monitoring_enabled = False

# --- Matplotlib Figure Setup for LIVE Plotting ---
plt.style.use('classic')
fig_live, ax_live = plt.subplots(figsize=(10, 6))
fig_live.canvas.manager.set_window_title('Real-time System Performance Metrics')

line_cpu, = ax_live.plot([], [], label='CPU Usage (%)', color='blue')
line_ram, = ax_live.plot([], [], label='RAM Usage (%)', color='magenta')
line_gpu, = ax_live.plot([], [], label='GPU Usage (%)', color='red')
line_vram, = ax_live.plot([], [], label='VRAM Usage (%)', color='orange')
line_yolo_conf, = ax_live.plot([], [], label='YOLO Confidence (%)', color='green') # Perubahan label
line_fps, = ax_live.plot([], [], label='FPS', color='cyan')

ax_live.set_ylim(0, 100)
ax_live.set_xlabel('Time (seconds)')
ax_live.set_ylabel('Percentage / Value')
ax_live.set_title('Real-time System Performance')
ax_live.legend(loc='upper left')
ax_live.grid(True)

plt.ion() # Turn on interactive mode for live plotting
plt.show(block=False) # Show non-blocking

def calculate_and_print_averages(cpu_data, ram_data, gpu_data, vram_data, yolo_confidence_data, fps_data): # Perubahan parameter
    print("\n--- Average Performance Metrics ---")

    # Calculate average for all metrics
    def calculate_avg(data):
        return sum(data) / len(data) if data else 0

    avg_cpu = calculate_avg(cpu_data)
    avg_ram = calculate_avg(ram_data)
    avg_gpu = calculate_avg(gpu_data)
    avg_vram = calculate_avg(vram_data)
    avg_fps = calculate_avg(fps_data)

    # For YOLO Confidence, filter out 0 values before calculating average
    filtered_yolo_conf = [conf for conf in yolo_confidence_data if conf > 0] # Perubahan variabel
    avg_yolo_conf = calculate_avg(filtered_yolo_conf) # Perubahan variabel

    print(f"Average CPU Usage: {avg_cpu:.2f}%")
    print(f"Average RAM Usage: {avg_ram:.2f}%")
    if gpu_data:
        print(f"Average GPU Usage: {avg_gpu:.2f}%")
        print(f"Average VRAM Usage: {avg_vram:.2f}%")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"Average YOLO Confidence (excluding 0): {avg_yolo_conf:.2f}%") # Perubahan teks
    print("-----------------------------------")

    # --- Tambahan Kode untuk Membuat Grafik ---

    # Siapkan data untuk grafik
    labels = ['CPU Usage', 'RAM Usage', 'GPU Usage', 'VRAM Usage', 'FPS', 'YOLO Confidence'] # Perubahan label
    averages = [avg_cpu, avg_ram, avg_gpu, avg_vram, avg_fps, avg_yolo_conf] # Perubahan variabel
    colors = ['skyblue', 'lightgreen', 'salmon', 'gold', 'purple', 'magenta']

    # Buat grafik batang
    plt.figure(figsize=(10, 6))
    plt.bar(labels, averages, color=colors)
    plt.ylabel('Average Value (%)')
    plt.title('Average Performance Metrics Comparison')
    plt.xticks(rotation=45, ha='right')  # Putar label sumbu-x agar mudah dibaca
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout() # Sesuaikan layout agar tidak ada yang terpotong

    # Tampilkan nilai di atas setiap batang
    for i, v in enumerate(averages):
        plt.text(i, v + 1, f"{v:.2f}", ha='center', va='bottom')

    # Tampilkan grafik
    plt.show()

# --- Function to plot all data in one window at the end ---
def plot_all_metrics_combined_at_end(timestamps, cpu_data, ram_data, gpu_data, vram_data, yolo_confidence_data, fps_data): # Perubahan parameter
    print("\nGenerating final combined performance plots...")

    metrics = {
        "CPU Usage": (cpu_data, '%', 'blue'),
        "GPU Usage": (gpu_data, '%', 'red'),
        "RAM Usage": (ram_data, '%', 'magenta'),
        "VRAM Usage": (vram_data, '%', 'orange'),
        "FPS": (fps_data, 'FPS', 'cyan')
    }

    # Create a single figure with a grid of subplots (e.g., 2 rows, 3 columns)
    fig_final, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
    fig_final.suptitle('Overall System Performance Summary', color='white', fontsize=16)

    # Flatten the axes array for easy iteration if using 2D grid
    axes = axes.flatten()

    for i, (name, (data, unit, color)) in enumerate(metrics.items()):
        ax = axes[i]
        
        # Set dark theme for each subplot
        ax.set_facecolor('white')
        ax.spines['bottom'].set_color('black')
        ax.spines['top'].set_color('black')
        ax.spines['left'].set_color('black')
        ax.spines['right'].set_color('black')
        ax.xaxis.label.set_color('black')
        ax.yaxis.label.set_color('black')
        ax.tick_params(axis='x', colors='black')
        ax.tick_params(axis='y', colors='black')
        ax.title.set_color('black')
        ax.grid(True, color='gray', linestyle='--')

        if not data:
            ax.text(0.5, 0.5, "No Data", horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, color='gray', fontsize=14)
            ax.set_title(f'{name}', color='black')
            ax.set_xlabel('Time (s)', color='black')
            ax.set_ylabel(f'{unit}', color='black')
            continue

        ax.plot(timestamps, data, label=f'{name} ({unit})', color=color)
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel(f'{name} ({unit})')
        ax.set_title(f'{name} Over Time')
        
        # Adjust y-limits for percentages and FPS
        if '%' in unit:
            ax.set_ylim(0, 100)
        else: # For FPS
            ax.set_ylim(bottom=0)

    # Adjust layout to prevent overlap and set overall figure background
    fig_final.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_final.set_facecolor('white')

    # Display the combined figure and wait for it to be closed
    plt.ioff()
    plt.show()

    print("Final combined plots window closed. Program will now terminate.")

def simple_bbox_overlap(box1, box2):
    """Calculate overlap ratio between two boxes"""
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection = (x_right - x_left) * (y_bottom - y_top)
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    return intersection / max(area_box1, 1e-6)

start_program_time = time.time()

while True:
    ret, frame = camera.read()
    current_time = time.time()
    relative_time = current_time - start_program_time

    if not ret:
        print("Failed to read frame")
        break

    frame_count += 1

    # FPS calculation
    if current_time - last_fps_time >= 1.0:
        fps = frame_count / (current_time - last_fps_time)
        frame_count = 0
        last_fps_time = current_time

    # YOLO detection every frame (GPU accelerated)
    results = model(frame, verbose=False)

    current_ids = []
    current_frame_yolo_confidences = [] # Perubahan nama variabel
    face_recognition_performed = False # Tambahkan flag untuk mencegah double hitung akurasi

    # Process detections
    for result in results[0].boxes:
        x1, y1, x2, y2 = result.xyxy[0].cpu().numpy()
        conf = result.conf[0].item()
        cls = result.cls[0].item()

        if conf > 0.3 and model.names[int(cls)] == "person":
            # --- Mencatat Akurasi YOLO ---
            # Kumpulkan semua confidence score dari deteksi "person"
            current_frame_yolo_confidences.append(conf * 100) # Kalikan 100 untuk mendapatkan persen

            new_bbox = (x1, y1, x2, y2)
            matched_id = None
            best_overlap = 0

            for person_id, track in active_person_tracks.items():
                if current_time - track['last_seen'] > 1.0:
                    continue
                overlap = simple_bbox_overlap(new_bbox, track['last_bbox'])
                if overlap > 0.5 and overlap > best_overlap:
                    best_overlap = overlap
                    matched_id = person_id

            if matched_id:
                person_id = matched_id
                active_person_tracks[person_id]['last_bbox'] = new_bbox
                active_person_tracks[person_id]['last_seen'] = current_time
            else:
                person_id = next_person_id
                while person_id in used_person_ids:
                    person_id += 1
                used_person_ids.add(person_id)
                next_person_id = person_id + 1
                active_person_tracks[person_id] = {
                    'last_bbox': new_bbox,
                    'last_seen': current_time
                }

            current_ids.append(person_id)

            # Bagian Face Recognition (dapat tetap ada, tetapi tidak memengaruhi grafik akurasi)
            if current_time - last_face_recog_time > face_recognition_interval and person_id not in id_tags:
                face_frame = frame[int(y1):int(y2), int(x1):int(x2)]
                if face_frame.size == 0:
                    continue

                rgb_face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
                face_encodings = face_recognition.face_encodings(rgb_face_frame)

                if face_encodings:
                    matches = face_recognition.compare_faces(known_face_encodings, face_encodings[0])
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encodings[0])
                    best_match_index = np.argmin(face_distances)
                    similarity = (1 - face_distances[best_match_index]) * 100
                    
                    if matches[best_match_index] and similarity >= 60:
                        id_tags[person_id] = known_face_names[best_match_index]
                        name = known_face_names[best_match_index]
                        if name != "Unknown" and name not in person_presence_start_time:
                            person_presence_start_time[name] = current_time
                            person_total_presence_time.setdefault(name, 0)
                    else:
                        pass
                
                last_face_recog_time = current_time
                face_recognition_performed = True

            name = id_tags.get(person_id, "Unknown")
            display_text = f"ID {person_id} | {name}"
            
            if name != "Unknown":
                if name in person_presence_start_time:
                    session_duration = current_time - person_presence_start_time[name]
                    total_seconds = int(person_total_presence_time[name] + session_duration)
                    hours = total_seconds // 3600
                    minutes = (total_seconds % 3600) // 60
                    seconds = total_seconds % 60
                    time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                    display_text += f" ({time_str})"
                else:
                    display_text += " (Tracking...)"

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, display_text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    for name, start_time in list(person_presence_start_time.items()):
        if name != "Unknown":
            found_in_current_frame = False
            for pid in current_ids:
                if id_tags.get(pid) == name:
                    found_in_current_frame = True
                    break

            if not found_in_current_frame:
                if name in person_presence_start_time:
                    session_duration = current_time - person_presence_start_time[name]
                    person_total_presence_time[name] += session_duration
                    del person_presence_start_time[name]

    expired_ids = [pid for pid, track in active_person_tracks.items()
                   if current_time - track['last_seen'] > 5.0]
    for pid in expired_ids:
        if pid in id_tags:
            name_of_expired_person = id_tags[pid]
            if name_of_expired_person != "Unknown" and name_of_expired_person in person_presence_start_time:
                session_duration = current_time - person_presence_start_time[name_of_expired_person]
                person_total_presence_time[name_of_expired_person] += session_duration
                del person_presence_start_time[name_of_expired_person]
            del id_tags[pid]
        del active_person_tracks[pid]

    # --- Real-time System Monitoring and Data Logging ---
    if current_time - last_monitor_update_time >= monitor_update_interval:
        # 1. CPU Usage
        cpu_percent = psutil.cpu_percent(interval=None)
        cpu_usage_live.append(cpu_percent)
        all_cpu_usage.append(cpu_percent)

        # 2. RAM Usage
        ram_percent = psutil.virtual_memory().percent
        ram_usage_live.append(ram_percent)
        all_ram_usage.append(ram_percent)

        # 3. GPU Usage & 4. VRAM Usage (NVIDIA only)
        if gpu_monitoring_enabled:
            try:
                utilization = nvmlDeviceGetUtilizationRates(handle)
                gpu_usage = utilization.gpu
                gpu_usage_live.append(gpu_usage)
                all_gpu_usage.append(gpu_usage)

                memory_info = nvmlDeviceGetMemoryInfo(handle)
                used_vram_mb = memory_info.used / (1024 * 1024)
                vram_percent = (used_vram_mb / VRAM_CAPACITY_MB) * 100 if VRAM_CAPACITY_MB > 0 else 0
                vram_usage_live.append(vram_percent)
                all_vram_usage.append(vram_percent)
            except NVMLError as error:
                print(f"Error getting GPU info: {error}")
                gpu_usage_live.append(0)
                all_gpu_usage.append(0)
                vram_usage_live.append(0)
                all_vram_usage.append(0)
        else:
            gpu_usage_live.append(0)
            all_gpu_usage.append(0)
            vram_usage_live.append(0)
            all_vram_usage.append(0)

        # 5. Akurasi (sekarang mencatat rata-rata confidence YOLO)
        if current_frame_yolo_confidences:
            avg_yolo_confidence = sum(current_frame_yolo_confidences) / len(current_frame_yolo_confidences)
            yolo_confidence_live.append(avg_yolo_confidence)
            all_yolo_confidence.append(avg_yolo_confidence)
        else:
            yolo_confidence_live.append(0)
            all_yolo_confidence.append(0)

        # 6. FPS
        fps_live.append(fps)
        all_fps.append(fps)

        # Time for X-axis (relative to start of program)
        time_live.append(relative_time)
        all_time.append(relative_time)

        # Update the Matplotlib LIVE plot
        line_cpu.set_data(time_live, cpu_usage_live)
        line_ram.set_data(time_live, ram_usage_live)
        line_gpu.set_data(time_live, gpu_usage_live)
        line_vram.set_data(time_live, vram_usage_live)
        line_yolo_conf.set_data(time_live, yolo_confidence_live) # Perubahan variabel
        line_fps.set_data(time_live, fps_live)

        ax_live.set_xlim(max(0, time_live[-1] - MAX_LIVE_DATA_POINTS * monitor_update_interval), time_live[-1] + monitor_update_interval)

        max_y_limit = max(100, max(fps_live) if fps_live else 0, max(cpu_usage_live) if cpu_usage_live else 0, max(ram_usage_live) if ram_usage_live else 0)
        ax_live.set_ylim(0, max_y_limit * 1.1)

        fig_live.canvas.draw()
        fig_live.canvas.flush_events()

        last_monitor_update_time = current_time
        current_frame_yolo_confidences = [] # Reset list untuk frame berikutnya

    # Get the last value of each deque to display on the OpenCV window
    cpu_val = cpu_usage_live[-1] if cpu_usage_live else 0
    ram_val = ram_usage_live[-1] if ram_usage_live else 0
    gpu_val = gpu_usage_live[-1] if gpu_usage_live else 0
    vram_val = vram_usage_live[-1] if vram_usage_live else 0
    yolo_conf_val = yolo_confidence_live[-1] if yolo_confidence_live else 0 # Perubahan variabel

    # Display system info on OpenCV window
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"CPU: {cpu_val:.1f}%", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"RAM: {ram_val:.1f}%", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    if gpu_monitoring_enabled:
        cv2.putText(frame, f"GPU: {gpu_val:.1f}%", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"VRAM: {vram_val:.1f}%", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.putText(frame, f"YOLO Conf: {yolo_conf_val:.1f}%", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2) # Perubahan label dan variabel

    y_offset = 210
    cv2.putText(frame, "Presence:", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y_offset += 30
    for name, total_seconds in person_total_presence_time.items():
        if name != "Unknown":
            current_session_duration = 0
            for pid, person_name in id_tags.items():
                if person_name == name and name in person_presence_start_time:
                    current_session_duration = current_time - person_presence_start_time[name]
                    break
            
            display_total_seconds = int(total_seconds + current_session_duration)
            hours = display_total_seconds // 3600
            minutes = (display_total_seconds % 3600) // 60
            seconds = display_total_seconds % 60
            time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            cv2.putText(frame, f"- {name}: {time_str}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y_offset += 25
            
    # Display the frame in the separate OpenCV window
    cv2.imshow("Attendance System", frame)
    
    # Check for 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup and Final Plotting ---
camera.release()
cv2.destroyAllWindows()
nvmlShutdown()
plt.close(fig_live)

# Call the function to generate and display combined plots
plot_all_metrics_combined_at_end(all_time, all_cpu_usage, all_ram_usage, all_gpu_usage, all_vram_usage, all_yolo_confidence, all_fps) # Perubahan variabel
calculate_and_print_averages(all_cpu_usage, all_ram_usage, all_gpu_usage, all_vram_usage, all_yolo_confidence, all_fps) # Perubahan variabel

print("Program finished.")
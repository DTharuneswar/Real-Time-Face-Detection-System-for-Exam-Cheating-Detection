from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
import cv2
import numpy as np
import threading
import base64
import time
import mediapipe as mp
import asyncio
from typing import Optional
from contextlib import contextmanager
import csv
import os
from datetime import datetime
from pathlib import Path

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

class FaceDetectionState:
    def __init__(self):
        self.detected: bool = False
        self.looking_at_camera: bool = False
        self.confidence: float = 0.0
        self.head_rotation: float = 0.0
        self.multiple_faces: bool = False
        self._lock = threading.Lock()
    
    @contextmanager
    def update(self):
        with self._lock:
            yield self
    
    def to_dict(self):
        return {
            "detected": self.detected,
            "looking_at_camera": self.looking_at_camera,
            "confidence": self.confidence,
            "head_rotation": self.head_rotation,
            "multiple_faces": self.multiple_faces
        }

face_state = FaceDetectionState()
frame_data: Optional[str] = None
frame_lock = threading.Lock()
current_suc_id = None

# Ensure logs directory exists
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)
csv_file = LOGS_DIR / "looking_away_logs.csv"

# Initialize CSV file with headers if it doesn't exist
if not csv_file.exists():
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['timestamp', 'suc_id'])

csv_lock = threading.Lock()

def log_looking_away(suc_id):
    if not suc_id:
        return
    
    try:
        with csv_lock:
            with open(csv_file, 'a', newline='') as file:
                writer = csv.writer(file)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                writer.writerow([timestamp, suc_id])
    except Exception as e:
        print(f"Error writing to CSV: {e}")

def calculate_face_metrics(landmarks, image_width: int):
    landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    
    # Key facial landmarks
    left_eye_indices = [33, 133, 157, 158]
    right_eye_indices = [362, 263, 384, 385]
    nose_indices = [1, 4, 19]
    
    # Calculate positions
    left_eye_x = np.mean(landmarks_array[left_eye_indices, 0]) * image_width
    right_eye_x = np.mean(landmarks_array[right_eye_indices, 0]) * image_width
    nose_x = np.mean(landmarks_array[nose_indices, 0]) * image_width
    
    # Calculate depth and rotation
    left_eye_z = np.mean(landmarks_array[left_eye_indices, 2])
    right_eye_z = np.mean(landmarks_array[right_eye_indices, 2])
    head_rotation = np.arctan2(right_eye_z - left_eye_z, right_eye_x - left_eye_x) * 180 / np.pi
    
    # Calculate confidence
    z_values = landmarks_array[[*left_eye_indices, *right_eye_indices], 2]
    confidence = min(1.0, max(0.0, 1.0 - np.std(z_values) * 10))
    
    return {
        'left_eye_x': float(left_eye_x),
        'right_eye_x': float(right_eye_x),
        'nose_x': float(nose_x),
        'head_rotation': float(head_rotation),
        'confidence': float(confidence)
    }

def is_looking_at_camera(metrics, image_width: int) -> bool:
    horizontal_deviation = abs(metrics['nose_x'] - image_width / 2)
    max_deviation = image_width * 0.15
    eye_symmetry = abs((metrics['left_eye_x'] - metrics['nose_x']) + (metrics['right_eye_x'] - metrics['nose_x']))
    max_asymmetry = image_width * 0.1
    max_rotation = 25.0
    
    return (metrics['confidence'] >= 0.7 and
            horizontal_deviation < max_deviation and 
            eye_symmetry < max_asymmetry and 
            abs(metrics['head_rotation']) < max_rotation)

def capture_video():
    global frame_data
    
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=5,
        refine_landmarks=True,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
        
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue
                
            frame = cv2.resize(frame, (640, 480))
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)
            
            with face_state.update() as state:
                if results.multi_face_landmarks:
                    num_faces = len(results.multi_face_landmarks)
                    state.multiple_faces = num_faces > 1
                    face_landmarks = results.multi_face_landmarks[0]
                    metrics = calculate_face_metrics(face_landmarks.landmark, frame.shape[1])
                    state.detected = True
                    state.looking_at_camera = is_looking_at_camera(metrics, frame.shape[1])
                    state.confidence = metrics['confidence']
                    state.head_rotation = metrics['head_rotation']
                else:
                    state.detected = False
                    state.looking_at_camera = False
                    state.confidence = 0.0
                    state.head_rotation = 0.0
                    state.multiple_faces = False

            if state.detected and not state.looking_at_camera and current_suc_id:
                cv2.putText(frame, "Please Look at Camera", (20, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                log_looking_away(current_suc_id)

            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            with frame_lock:
                frame_data = base64.b64encode(buffer).decode('utf-8')
                
            time.sleep(0.033)  # ~30 FPS
            
    except Exception as e:
        print(f"Error in video capture: {e}")
    finally:
        cap.release()
        face_mesh.close()

# Start video capture thread
video_thread = threading.Thread(target=capture_video, daemon=True)
video_thread.start()

@app.get("/")
async def get():
    return FileResponse("static/index.html")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global current_suc_id
    await websocket.accept()
    
    try:
        # Get SUC ID from first message
        current_suc_id = await websocket.receive_text()
        print(f"Connected user: {current_suc_id}")
        
        while True:
            await asyncio.sleep(0.1)  # Prevent busy-waiting
            with frame_lock:
                if frame_data is not None:
                    await websocket.send_json({
                        'frame': frame_data,
                        'metrics': face_state.to_dict()
                    })
    except WebSocketDisconnect:
        print(f"Client disconnected: {current_suc_id}")
        current_suc_id = None
    except Exception as e:
        print(f"WebSocket error: {e}")
        current_suc_id = None

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

import cv2
import numpy as np
import asyncio
import json
import gc
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import base64
from ultralytics import YOLO
from pathlib import Path

app = FastAPI()

# Enable CORS for mobile access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AppState:
    def __init__(self):
        self.confidence = 0.4
        self.iou = 0.7
        self.model_type = 'general'
        self.show_edge_detection = False
        self.camera_brightness = 0.0
        self.camera_contrast = 1.0
        self.camera_saturation = 1.0
        self.camera_exposure = 0.0
        self.model = None
        self.model_lock = asyncio.Lock()
        self.last_model_type = None

state = AppState()

class ConnectionManager:
    def __init__(self):
        self.active_connections: dict = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]

    async def send_to_client(self, client_id: str, message: str):
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(message)
            except:
                self.disconnect(client_id)

manager = ConnectionManager()

def adjust_frame_lighting(frame: np.ndarray, brightness: float, contrast: float, 
                         saturation: float, exposure: float) -> np.ndarray:
    """Adjust frame lighting."""
    img = frame.astype(np.float32) / 255.0
    
    exposure_factor = 2.0 ** exposure
    img = img * exposure_factor
    
    brightness_factor = brightness / 100.0
    img = img + brightness_factor
    
    img = 0.5 + contrast * (img - 0.5)
    
    if saturation != 1.0:
        hsv = cv2.cvtColor((np.clip(img, 0, 1) * 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = hsv[:, :, 1] * saturation
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0
    
    img = np.clip(img, 0, 1) * 255
    return img.astype(np.uint8)
async def load_model_if_needed(model_type: str):
    """Load model ONLY if type changed (NOT on every settings update)."""
    if state.last_model_type == model_type and state.model is not None:
        return  # Model already loaded, skip!
    
    async with state.model_lock:
        try:
            # Release old model
            if state.model is not None:
                del state.model
                gc.collect()
            
            if model_type == 'face':
                print("Loading face detection model...")
                state.model = YOLO('yolov8n-face.pt')
            else:
                print("Loading general detection model...")
                state.model = YOLO('yolov8n.pt')
            
            state.last_model_type = model_type
            print(f"✓ Model loaded: {model_type}")
        
        except Exception as e:
            print(f"Model loading error: {e}")
            state.model = None
def get_class_color(class_id: int) -> tuple:
    """Get unique BGR color for each class."""
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 255), (0, 128, 255),
        (255, 128, 0), (128, 255, 0), (0, 255, 128), (255, 0, 128),
    ]
    return colors[class_id % len(colors)]

def process_frame(frame):
    """Process frame with YOLO."""
    try:
        if state.model is None:
            return frame
        
        # Apply camera adjustments
        if (state.camera_brightness != 0.0 or state.camera_contrast != 1.0 or 
            state.camera_saturation != 1.0 or state.camera_exposure != 0.0):
            frame = adjust_frame_lighting(frame, state.camera_brightness, 
                                        state.camera_contrast, state.camera_saturation, 
                                        state.camera_exposure)
        
        yolo_img = frame.copy()
        
        # YOLO Detection
        if state.model is not None:
            try:
                results = state.model.track(
                    frame,
                    conf=float(state.confidence),
                    iou=float(state.iou),
                    verbose=False
                )
                
                if results and len(results) > 0:
                    result = results[0]
                    if result.boxes is not None and len(result.boxes) > 0:
                        for box in result.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            track_id = int(box.id) if box.id is not None else -1
                            conf = float(box.conf)
                            cls = int(box.cls) if box.cls is not None else -1
                            
                            class_name = state.model.names.get(cls, f"Class{cls}") if hasattr(state.model, 'names') else f"Class{cls}"
                            class_color = get_class_color(cls)
                            
                            # Draw bounding box
                            cv2.rectangle(yolo_img, (x1, y1), (x2, y2), class_color, 2)
                            
                            # Draw track ID
                            if track_id >= 0:
                                cv2.putText(yolo_img, f"ID:{track_id}", (x1, y1 - 10),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, class_color, 2)
                            
                            # Draw class and confidence
                            cv2.putText(yolo_img, f"{class_name} {conf:.2f}", (x1, y2 + 20),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, class_color, 2)
                            
                            # Edge detection
                            if state.show_edge_detection:
                                box_region = frame[y1:y2, x1:x2]
                                if box_region.size > 0:
                                    gray = cv2.cvtColor(box_region, cv2.COLOR_BGR2GRAY)
                                    edges = cv2.Canny(gray, 50, 150)
                                    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                                    for contour in contours:
                                        offset_contour = contour + np.array([x1, y1])
                                        cv2.drawContours(yolo_img, [offset_contour], 0, (0, 255, 255), 1)
            
            except Exception as e:
                print(f"Detection error: {e}")
        
        return yolo_img
    
    except Exception as e:
        print(f"Frame processing error: {e}")
        return frame

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    
    try:
        while True:
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                
                if message['type'] == 'settings':
                    # Update settings
                    settings = message['data']
                    state.confidence = settings.get('confidence', state.confidence)
                    state.iou = settings.get('iou', state.iou)
                    state.show_edge_detection = settings.get('show_edge_detection', state.show_edge_detection)
                    state.camera_brightness = settings.get('camera_brightness', state.camera_brightness)
                    state.camera_contrast = settings.get('camera_contrast', state.camera_contrast)
                    state.camera_saturation = settings.get('camera_saturation', state.camera_saturation)
                    state.camera_exposure = settings.get('camera_exposure', state.camera_exposure)
                    
                    # ONLY load model if TYPE changed
                    new_model_type = settings.get('model_type', state.model_type)
                    if new_model_type != state.model_type:
                        state.model_type = new_model_type
                        await load_model_if_needed(new_model_type)
                
                elif message['type'] == 'frame':
                    # Ensure model is loaded
                    await load_model_if_needed(state.model_type)
                    
                    if state.model is None:
                        continue
                    
                    # Process frame from mobile camera
                    frame_data = message['data']
                    frame_bytes = base64.b64decode(frame_data)
                    nparr = np.frombuffer(frame_bytes, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if frame is not None:
                        # Resize for performance
                        frame = cv2.resize(frame, (1280, 720))
                        
                        # Process
                        processed = process_frame(frame)
                        
                        # Encode back
                        _, buffer = cv2.imencode('.jpg', processed, [cv2.IMWRITE_JPEG_QUALITY, 80])
                        result_b64 = base64.b64encode(buffer).decode()
                        
                        # Send ONLY to this client (not broadcast)
                        await manager.send_to_client(client_id, json.dumps({
                            'type': 'processed_frame',
                            'data': result_b64
                        }))
            
            except Exception as e:
                print(f"Message error: {e}")
    
    except WebSocketDisconnect:
        manager.disconnect(client_id)

@app.get("/")
async def get_index():
    return HTMLResponse(get_html())

def get_html():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover">
    <title>YOLO Real-Time Detection</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a1a;
            color: #fff;
            width: 100vw;
            height: 100vh;
            overflow: hidden;
        }
        
        .container {
            display: flex;
            flex-direction: column;
            width: 100%;
            height: 100%;
        }
        
        .video-section {
            flex: 1;
            position: relative;
            background: #000;
            overflow: hidden;
        }
        
        #videoCanvas {
            width: 100%;
            height: 100%;
            object-fit: contain;
        }
        
        #cameraFeed {
            position: absolute;
            width: 100%;
            height: 100%;
            object-fit: cover;
            display: none;
        }
        
        .controls-section {
            background: #111;
            border-top: 1px solid #333;
            padding: 15px;
            max-height: 40vh;
            overflow-y: auto;
        }
        
        .control-group {
            margin-bottom: 15px;
        }
        
        .control-group label {
            display: block;
            font-size: 13px;
            color: #aaa;
            margin-bottom: 5px;
            font-weight: 600;
        }
        
        .control-group input[type="range"] {
            width: 100%;
            height: 5px;
            border-radius: 3px;
            background: #333;
            outline: none;
            -webkit-appearance: none;
            accent-color: #667eea;
        }
        
        .control-group input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 16px;
            height: 16px;
            border-radius: 50%;
            background: #667eea;
            cursor: pointer;
        }
        
        .control-group select,
        .control-group button {
            width: 100%;
            padding: 10px;
            border: 1px solid #333;
            border-radius: 6px;
            background: #222;
            color: #fff;
            font-size: 14px;
            cursor: pointer;
        }
        
        .control-group button {
            background: #667eea;
            border: none;
            font-weight: 600;
            margin-top: 10px;
        }
        
        .control-group button:active {
            background: #5568d3;
        }
        
        .checkbox-group {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .checkbox-group input[type="checkbox"] {
            width: 18px;
            height: 18px;
            cursor: pointer;
            accent-color: #667eea;
        }
        
        .status {
            display: inline-block;
            padding: 8px 12px;
            background: #222;
            border-left: 3px solid #667eea;
            border-radius: 4px;
            font-size: 12px;
            margin-bottom: 15px;
        }
        
        .status.connected {
            border-left-color: #4caf50;
            color: #4caf50;
        }
        
        .status.disconnected {
            border-left-color: #f44336;
            color: #f44336;
        }
        
        @media (max-width: 800px) {
            .controls-section {
                max-height: 35vh;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="video-section">
            <video id="cameraFeed" playsinline autoplay muted></video>
            <canvas id="videoCanvas"></canvas>
        </div>
        
        <div class="controls-section">
            <div class="status" id="status">Connecting...</div>
            
            <div class="control-group">
                <button onclick="startCamera()">🎥 Start Camera</button>
                <button onclick="stopCamera()">⏹ Stop Camera</button>
            </div>
            
            <div class="control-group">
                <label>Model Type</label>
                <select id="modelType" onchange="updateSettings()">
                    <option value="general">General Objects</option>
                    <option value="face">Face Detection</option>
                </select>
            </div>
            
            <div class="control-group">
                <label>Confidence: <span id="confValue">0.40</span></label>
                <input type="range" id="confidence" min="0.1" max="1" step="0.05" value="0.4" oninput="updateSettings(); document.getElementById('confValue').textContent = this.value;">
            </div>
            
            <div class="control-group">
                <label>IoU Threshold: <span id="iouValue">0.70</span></label>
                <input type="range" id="iou" min="0.1" max="1" step="0.05" value="0.7" oninput="updateSettings(); document.getElementById('iouValue').textContent = this.value;">
            </div>
            
            <div class="control-group checkbox-group">
                <input type="checkbox" id="edgeDetection" onchange="updateSettings()">
                <label for="edgeDetection" style="margin: 0;">Edge Detection</label>
            </div>
            
            <div class="control-group">
                <label>Brightness: <span id="brightValue">0</span></label>
                <input type="range" id="brightness" min="-100" max="100" step="10" value="0" oninput="updateSettings(); document.getElementById('brightValue').textContent = this.value;">
            </div>
        </div>
    </div>
    
    <script>
        const clientId = 'client_' + Math.random().toString(36).substr(2, 9);
        const ws = new WebSocket(`ws://${window.location.host}/ws/${clientId}`);
        const video = document.getElementById('cameraFeed');
        const canvas = document.getElementById('videoCanvas');
        const ctx = canvas.getContext('2d');
        let stream = null;
        let isRunning = false;
        
        // Set canvas size
        function resizeCanvas() {
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight * 0.6;
        }
        resizeCanvas();
        window.addEventListener('resize', resizeCanvas);
        
        ws.onopen = () => {
            setStatus('Connected', true);
        };
        
        ws.onmessage = (event) => {
            try {
                const message = JSON.parse(event.data);
                
                if (message.type === 'processed_frame') {
                    const img = new Image();
                    img.onload = () => {
                        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                    };
                    img.src = 'data:image/jpeg;base64,' + message.data;
                }
            } catch (e) {
                console.error('Message error:', e);
            }
        };
        
        ws.onerror = () => setStatus('Connection Error', false);
        ws.onclose = () => setStatus('Disconnected', false);
        
        function setStatus(text, connected) {
            const status = document.getElementById('status');
            status.textContent = text;
            status.className = 'status ' + (connected ? 'connected' : 'disconnected');
        }
        
        async function startCamera() {
            try {
                stream = await navigator.mediaDevices.getUserMedia({
                    video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } },
                    audio: false
                });
                video.srcObject = stream;
                isRunning = true;
                captureFrames();
            } catch (err) {
                alert('Camera access denied: ' + err.message);
            }
        }
        
        function stopCamera() {
            isRunning = false;
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
                stream = null;
            }
        }
        
        function captureFrames() {
            if (!isRunning || !stream) return;
            
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            ctx.drawImage(video, 0, 0);
            
            canvas.toBlob((blob) => {
                const reader = new FileReader();
                reader.onloadend = () => {
                    const base64 = reader.result.split(',')[1];
                    if (ws.readyState === WebSocket.OPEN) {
                        ws.send(JSON.stringify({
                            type: 'frame',
                            data: base64
                        }));
                    }
                };
                reader.readAsDataURL(blob);
            }, 'image/jpeg', 0.7);
            
            setTimeout(captureFrames, 100); // ~10 FPS
        }
        
        function updateSettings() {
            const settings = {
                confidence: parseFloat(document.getElementById('confidence').value),
                iou: parseFloat(document.getElementById('iou').value),
                model_type: document.getElementById('modelType').value,
                show_edge_detection: document.getElementById('edgeDetection').checked,
                camera_brightness: parseFloat(document.getElementById('brightness').value),
                camera_contrast: 1.0,
                camera_saturation: 1.0,
                camera_exposure: 0.0
            };
            
            if (ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({
                    type: 'settings',
                    data: settings
                }));
            }
        }
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

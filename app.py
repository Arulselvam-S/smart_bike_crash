from flask import Flask, render_template, Response
import cv2
import numpy as np
from ultralytics import YOLO
from flask_socketio import SocketIO, emit

# Initialize Flask app and SocketIO
app = Flask(__name__)
socketio = SocketIO(app)

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Ensure you have this model file

# Initialize webcam
cap = cv2.VideoCapture(0)  # 0 for default webcam

# Setup counters and states
inside_count = 0
PASSENGER_LIMIT = 5  # Example passenger limit

# Function to generate frames for the webcam feed
def generate_frames():
    global inside_count

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLOv8 model
        results = model(frame)

        # Process detection and count people
        for detection in results[0].boxes.data:
            class_id = int(detection[5])  # Class ID
            confidence = float(detection[4])  # Confidence score

            if class_id == 0 and confidence > 0.4:  # Person detected
                x1, y1, x2, y2 = map(int, detection[:4])
                # Update count based on some condition, e.g., entering/exiting

        # Emit updates to frontend (e.g., inside_count and light status)
        socketio.emit('update_count', {'count': inside_count, 'overload': inside_count > PASSENGER_LIMIT})

        # Encode the frame in JPEG format to send to the browser
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Route for the webpage
@app.route('/')
def index():
    return render_template('index.html')

# Route to stream webcam feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Start the Flask server
if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000)

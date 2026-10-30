import cv2
import numpy as np
from ultralytics import YOLO
import serial
import time

# Setup serial connection (adjust COM port as needed)
ser = serial.Serial('COM3', 9600, timeout=1)
time.sleep(2)  # Wait for Arduino to reset
print("Serial connection established.")

# Load YOLO model
model = YOLO("yolov8n.pt")  # Adjust path if needed

# Open camera
cap = cv2.VideoCapture(0)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

line_x = frame_width // 2

tracked_objects = {}
next_object_id = 1
inside_count = 0

PASSENGER_LIMIT = 0  # Set max allowed passengers

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    current_frame_objects = {}

    for detection in results[0].boxes.data:
        class_id = int(detection[5])
        confidence = float(detection[4])

        if class_id == 0 and confidence > 0.4:
            x1, y1, x2, y2 = map(int, detection[:4])
            person_center_x = (x1 + x2) // 2

            assigned_id = None
            for object_id, prev_x in tracked_objects.items():
                if abs(person_center_x - prev_x) < 50:
                    assigned_id = object_id
                    break

            if assigned_id is None:
                assigned_id = next_object_id
                next_object_id += 1

            current_frame_objects[assigned_id] = person_center_x

            if assigned_id in tracked_objects:
                prev_x = tracked_objects[assigned_id]

                if prev_x < line_x and person_center_x >= line_x:
                    inside_count += 1
                    print(f"Person {assigned_id} ENTERED the room")
                elif prev_x > line_x and person_center_x <= line_x:
                    inside_count -= 1
                    inside_count = max(inside_count, 0)
                    print(f"Person {assigned_id} EXITED the room")

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {assigned_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    tracked_objects = current_frame_objects

    # Serial communication to Arduino
    if inside_count > PASSENGER_LIMIT:
        print("Sending: 1")
        ser.write(b'1')
    else:
        print("Sending: 0")
        ser.write(b'0')

    # Overlay info
    if inside_count > PASSENGER_LIMIT:
        cv2.putText(frame, "🚨 Overload Detected!", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.putText(frame, f"People Inside: {inside_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.line(frame, (line_x, 0), (line_x, frame_height), (0, 0, 255), 2)
    cv2.imshow("People Counting", frame)

    # Small delay to prevent flooding serial port
    time.sleep(0.1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
ser.close()
print("Program ended.")


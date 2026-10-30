import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 model (Ensure you have "yolov8n.pt" in the same directory or use the full path)
model = YOLO("yolov8n.pt")

# Open laptop camera
cap = cv2.VideoCapture(0)

# Get frame width and height
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define the vertical central line position (middle of the frame) - entry/exit point
line_x = frame_width // 2  

# Store object positions across frames for tracking
tracked_objects = {}  # {object_id: last_x_position}
next_object_id = 1  # ID counter for new detections

# Counters
inside_count = 0  # Number of people inside the room

# Set Passenger Limit
PASSENGER_LIMIT = 3  # Set the max number of passengers allowed

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 on the frame
    results = model(frame)

    # Store detected people and their positions in the current frame
    current_frame_objects = {}

    for detection in results[0].boxes.data:
        class_id = int(detection[5])  # Class ID
        confidence = float(detection[4])  # Confidence score

        # Class ID 0 corresponds to "person"
        if class_id == 0 and confidence > 0.4:
            x1, y1, x2, y2 = map(int, detection[:4])  # Bounding box coordinates
            person_center_x = (x1 + x2) // 2  # Get X-coordinate of person's center

            # Check if this person was already tracked in the last frame
            assigned_id = None
            for object_id, prev_x in tracked_objects.items():
                if abs(person_center_x - prev_x) < 50:  # If close to previous position, it's the same person
                    assigned_id = object_id
                    break

            if assigned_id is None:
                assigned_id = next_object_id
                next_object_id += 1

            current_frame_objects[assigned_id] = person_center_x

            # Check if they entered or exited the room
            if assigned_id in tracked_objects:
                prev_x = tracked_objects[assigned_id]

                if prev_x < line_x and person_center_x >= line_x:
                    inside_count += 1  # Entry detected
                    print(f"Person {assigned_id} ENTERED the room")
                elif prev_x > line_x and person_center_x <= line_x:
                    inside_count -= 1  # Exit detected
                    inside_count = max(inside_count, 0)  # Prevent negative count
                    print(f"Person {assigned_id} EXITED the room")

    # Update tracked objects for the next frame
    tracked_objects = current_frame_objects

    # Overload detection
    if inside_count > PASSENGER_LIMIT:
        print("🚨 Overload Detected! Passenger limit exceeded.")

    # Show the current frame with the number of people inside
    cv2.putText(frame, f"People Inside: {inside_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("People Counting", frame)

    # Break the loop if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()


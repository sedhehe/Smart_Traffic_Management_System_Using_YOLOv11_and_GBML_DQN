import cv2
from ultralytics import YOLO
from collections import defaultdict

# Load the YOLO model
model = YOLO('yolo11l.pt')
class_list = model.names

# Open the video file
video_capture = cv2.VideoCapture('test_videos/4.mp4')
red_line_y_position = 430  # Red line position

# Dictionary to store object counts by class
object_class_counts = defaultdict(int)
crossed_object_ids = set()  # Track object IDs that have crossed the line

while video_capture.isOpened():
    ret, frame = video_capture.read()
    if not ret:
        break

    # Run YOLO tracking on the frame
    results = model.track(frame, persist=True, classes=[1, 2, 3, 5, 6, 7])

    if results[0].boxes.data is not None:
        bounding_boxes = results[0].boxes.xyxy.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        class_indices = results[0].boxes.cls.int().cpu().tolist()
        confidences = results[0].boxes.conf.cpu()

        cv2.line(frame, (690, red_line_y_position), (1130, red_line_y_position), (0, 0, 255), 3)

        for box, track_id, class_idx, conf in zip(bounding_boxes, track_ids, class_indices, confidences):
            x1, y1, x2, y2 = map(int, box)
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            class_name = class_list[class_idx]

            cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)
            cv2.putText(frame, f"ID: {track_id} {class_name}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if center_y > red_line_y_position and track_id not in crossed_object_ids:
                crossed_object_ids.add(track_id)
                object_class_counts[class_name] += 1

            if class_name == 'emergency':
                print(f"Emergency vehicle detected: ID {track_id}")

        y_offset = 30
        for class_name, count in object_class_counts.items():
            cv2.putText(frame, f"{class_name}: {count}", (50, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            y_offset += 30

    cv2.imshow("YOLO Object Tracking & Counting", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
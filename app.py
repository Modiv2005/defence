from flask import Flask, Response
from ultralytics import YOLO
import cv2
import time

# ---------------- CONFIG ----------------
MODEL_PATH = "yolov8m.pt"
CONF_THRESHOLD = 0.5
THREAT_SOLDIER_COUNT = 3

CLASS_MAP = {
    "person": "Soldier",
    "car": "Military Vehicle",
    "truck": "Military Vehicle",
    "bus": "Military Vehicle"
}
# ----------------------------------------

app = Flask(__name__)

# Load model
model = YOLO(MODEL_PATH)

# Video source (0 = webcam, or replace with drone stream later)
cap = cv2.VideoCapture(0)

def process_frame(frame):
    results = model(frame, conf=CONF_THRESHOLD)[0]

    soldier_count = 0
    vehicle_count = 0

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]

        if label in CLASS_MAP:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            mapped_label = CLASS_MAP[label]

            if mapped_label == "Soldier":
                soldier_count += 1
                color = (0, 255, 0)
            else:
                vehicle_count += 1
                color = (255, 0, 0)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, mapped_label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return frame, soldier_count, vehicle_count


def generate_frames():
    prev_time = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame, soldiers, vehicles = process_frame(frame)

        # FPS calculation
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time

        # Display info
        cv2.putText(frame, f"Soldiers: {soldiers}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(frame, f"Vehicles: {vehicles}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(frame, f"FPS: {fps:.2f}", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        # Threat Alert
        if soldiers >= THREAT_SOLDIER_COUNT:
            cv2.putText(frame, "⚠ HIGH THREAT DETECTED", (20, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

        # Encode frame
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return """
    <html>
    <head>
        <title>Battlefield Surveillance</title>
        <style>
            body { background-color: #0f172a; color: white; text-align: center; }
            h1 { color: #38bdf8; }
            img { border: 3px solid #38bdf8; border-radius: 10px; }
        </style>
    </head>
    <body>
        <h1>AI Battlefield Surveillance System</h1>
        <img src="/video">
    </body>
    </html>
    """


@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)
from ultralytics import YOLO
import cv2
import time

# ---------------- CONFIG ----------------
MODEL_PATH = "yolov8m.pt"   # upgrade model
CONF_THRESHOLD = 0.5
THREAT_SOLDIER_COUNT = 3

CLASS_MAP = {
    "person": "Soldier",
    "car": "Military Vehicle",
    "truck": "Military Vehicle",
    "bus": "Military Vehicle"
}
# ----------------------------------------

def initialize_model():
    return YOLO(MODEL_PATH)

def process_frame(model, frame):
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
            else:
                vehicle_count += 1

            draw_box(frame, x1, y1, x2, y2, mapped_label)

    return frame, soldier_count, vehicle_count

def draw_box(frame, x1, y1, x2, y2, label):
    color = (0, 255, 0) if label == "Soldier" else (255, 0, 0)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def display_info(frame, soldiers, vehicles, fps):
    cv2.putText(frame, f"Soldiers: {soldiers}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(frame, f"Vehicles: {vehicles}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(frame, f"FPS: {fps:.2f}", (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    if soldiers >= THREAT_SOLDIER_COUNT:
        cv2.putText(frame, "⚠ HIGH THREAT", (20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

def main():
    model = initialize_model()
    cap = cv2.VideoCapture(0)

    prev_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame, soldiers, vehicles = process_frame(model, frame)

        # FPS calculation
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time

        display_info(frame, soldiers, vehicles, fps)

        cv2.imshow("Battlefield Surveillance", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
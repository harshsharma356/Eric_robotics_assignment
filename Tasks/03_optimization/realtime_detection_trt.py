import cv2
import time
from rfdetr import RFDETRInference

TRT_MODEL = "/home/galdrux/internship_projs/eric_robotics/technical_assignnment_ericRobotics/Tasks/03_optimization/trt/rfdetr-nano_fp16.trt"
CAMERA_INDEX = 0
CONFIDENCE_THRESHOLD = 0.65

print("[INFO] Loading TensorRT engine...")
model = RFDETRInference(TRT_MODEL)

print("[INFO] TensorRT engine loaded!")
print("[INFO] Classes:", model.class_names)

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

fps = 0
prev_time = time.time()

print("[INFO] TensorRT real-time detection started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("[ERROR] Failed to capture frame.")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    detections = model.predict(
        frame_rgb,
        threshold=CONFIDENCE_THRESHOLD
    )

    for i in range(len(detections)):
        x1, y1, x2, y2 = detections.xyxy[i].astype(int)
        confidence = float(detections.confidence[i])
        class_id = int(detections.class_id[i])
        class_name = model.class_names[class_id]

        label = f"{class_name} {confidence:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        (text_w, text_h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )

        cv2.rectangle(
            frame,
            (x1, max(0, y1 - text_h - 10)),
            (x1 + text_w, y1),
            (0, 255, 0),
            -1
        )

        cv2.putText(
            frame,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2
        )

    current_time = time.time()
    instantaneous_fps = 1 / max(current_time - prev_time, 1e-6)

    fps = 0.9 * fps + 0.1 * instantaneous_fps if fps else instantaneous_fps
    prev_time = current_time

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "RF-DETR Nano | TensorRT GPU",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.imshow("RF-DETR TensorRT Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("[INFO] TensorRT detection stopped.")
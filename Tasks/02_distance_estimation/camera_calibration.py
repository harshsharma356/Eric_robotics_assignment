import cv2
import time
import numpy as np
from rfdetr import RFDETR


CHECKPOINT = "/home/galdrux/internship_projs/eric_robotics/technical_assignnment_ericRobotics/model_training/checkpoint_best_total.pth"
CAMERA_INDEX = 0
CONFIDENCE_THRESHOLD = 0.65

cone_distance = 30#cm
cone_height = 6.5#cm
focal_length = None


print("[INFO] Loading trained RF-DETR model...")

model = RFDETR.from_checkpoint(CHECKPOINT)
predictor = model.inference(compile=False)

print("[INFO] Model loaded!")
print("[INFO] Classes:", predictor.class_names)

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

fps = 0
prev_time = time.time()

print("[INFO] Real-time detection started. Press 'q' to quit.")

while True:

    ret, frame = cap.read()

    if not ret:
        print("[ERROR] Failed to capture frame.")
        break

    h, w, c = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    detections = predictor.predict(
        frame_rgb,
        threshold=CONFIDENCE_THRESHOLD
    )

    cone_pixel_height = None

    for i in range(len(detections)):

        x1, y1, x2, y2 = detections.xyxy[i].astype(int)

        confidence = float(detections.confidence[i])
        class_id = int(detections.class_id[i])

        class_name = predictor.class_names[class_id]

        bbox_width = x2 - x1
        bbox_height = y2 - y1

        label = f"{class_name} {confidence:.2f}"

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        (text_w, text_h), _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            2
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
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2
        )

        if class_name == "cone":
            cone_pixel_height = bbox_height

    if cone_pixel_height is not None:

        cv2.putText(
            frame,
            f"Cone height in image: {cone_pixel_height}px",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

    if focal_length is None:

        cv2.putText(
            frame,
            "Calibration: Place cone at 30cm, press C",
            (20, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

    else:

        cv2.putText(
            frame,
            f"Focal Length: {focal_length:.2f} px",
            (20, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
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
        "RF-DETR Nano | Traffic Detection",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.imshow("RF-DETR Real-Time Detection", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("c"):

        if cone_pixel_height is not None:

            focal_length = cone_distance * cone_pixel_height / cone_height

            print(f"Focal length: {focal_length:.2f} pixels")

        else:
            print("[WARNING] No cone detected. Calibration failed.")

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("[INFO] Detection stopped.")
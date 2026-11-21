import cv2
import time
import numpy as np
from rfdetr import RFDETR


CHECKPOINT = "/home/galdrux/internship_projs/eric_robotics/technical_assignnment_ericRobotics/model_training/checkpoint_best_total.pth"
CAMERA_INDEX = 0
CONFIDENCE_THRESHOLD = 0.6


#Parameters for distance estimation
focal_lenght_pixels = 650#pixels
average_cone_height = 70#cm6.5
average_stop_sign_height = 60#cm
average_barrier_height = 90#cm




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

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    h,w,c = frame_rgb.shape


    detections = predictor.predict(
        frame_rgb,
        threshold=CONFIDENCE_THRESHOLD
        # device='cpu'
    )



#Parameters for distance estimation
    cone_pixel_height = None
    stop_sign_pixel_height = None
    barrier_pixel_height = None

    cone_distance = None
    barrier_distance = None
    stop_sign_distance = None

    for i in range(len(detections)):

        x1, y1, x2, y2 = detections.xyxy[i].astype(int)

        bbox_height = y2-y1
        bbox_width = x2-x1

        confidence = float(detections.confidence[i])
        class_id = int(detections.class_id[i])

        class_name = predictor.class_names[class_id]

        label = f"{class_name} {confidence:.2f}"

        # Bounding box
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
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2
        )

        if class_name == "cone":
            cone_pixel_height = bbox_height
            cone_distance = average_cone_height*focal_lenght_pixels/cone_pixel_height
            print(f"Cone estimated distance: {cone_distance}")

            cv2.putText(
                        frame,
                        f"Dist: {cone_distance:.2f}",
                        (x1, y2+50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2
                    )

        if class_name == "barrier":
            barrier_pixel_height = bbox_height
            barrier_distance = average_barrier_height*focal_lenght_pixels/barrier_pixel_height
            print(f"Barrier estimated Distance: {barrier_distance}")

            cv2.putText(
                        frame,
                        f"Dist: {barrier_distance:.2f}",
                        (x1, y2+50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2
                    )

        if class_name == "stop_sign":
            stop_sign_pixel_height = bbox_height
            stop_sign_distance = average_stop_sign_height*focal_lenght_pixels/stop_sign_pixel_height
            print(f"Stop Sign estimated Distance: {stop_sign_distance}")

            cv2.putText(
                        frame,
                        f"Dist: {stop_sign_distance:.2f}",
                        (x1, y2+50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
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

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("[INFO] Detection stopped.")
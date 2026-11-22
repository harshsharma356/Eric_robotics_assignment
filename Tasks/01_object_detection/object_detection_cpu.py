import cv2
import time
from rfdetr import RFDETRInference

CHECKPOINT = "/home/galdrux/internship_projs/eric_robotics/technical_assignnment_ericRobotics/model_training/checkpoint_best_total.pth"
CAMERA_INDEX = 0
CONFIDENCE_THRESHOLD = 0.65

print("[INFO] Loading ONNX model on CPU...")

model = RFDETRInference(CHECKPOINT, device="cpu")

print("[INFO] Model loaded!")
print("[INFO] Classes:", model.class_names)
print("[INFO] Runtime:", model.runtime_info)

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

fps = 0
prev_time = time.time()

# ================= INFERENCE BENCHMARK =================

print("\n[INFO] Starting inference benchmark...")

# Capture one frame to benchmark
ret, benchmark_frame = cap.read()

if not ret:
    raise RuntimeError("Failed to capture benchmark frame")

benchmark_frame_rgb = cv2.cvtColor(
    benchmark_frame,
    cv2.COLOR_BGR2RGB
)

# Warm-up runs
for _ in range(20):
    model.predict(
        benchmark_frame_rgb,
        threshold=CONFIDENCE_THRESHOLD
    )

# Actual benchmark
inference_times = []

for _ in range(100):
    start_time = time.perf_counter()

    model.predict(
        benchmark_frame_rgb,
        threshold=CONFIDENCE_THRESHOLD
    )

    end_time = time.perf_counter()

    inference_times.append(end_time - start_time)

# Calculate results
average_latency = (sum(inference_times) / len(inference_times)) * 1000
inference_fps = 1000 / average_latency

sorted_times = sorted(inference_times)
median_latency = sorted_times[len(sorted_times) // 2] * 1000

print("\n========== BENCHMARK RESULTS ==========")
print(f"Average latency : {average_latency:.2f} ms")
print(f"Median latency  : {median_latency:.2f} ms")
print(f"Inference FPS   : {inference_fps:.2f}")
print("=====================================\n")

# =====================================================

print("[INFO] CPU detection started. Press 'q' to quit.")

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
        "RF-DETR Nano | ONNX CPU",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.imshow("RF-DETR ONNX CPU Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("[INFO] CPU detection stopped.")
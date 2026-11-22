Eric Robotics Assignment


## Author

**Harsh Sharma**  
Robotics & Artificial Intelligence Engineering Student

### Contact Information
- **Phone:** +91-9867390027
- **Email:** harshsharma3561@gmail.com
- **GitHub:** https://github.com/harshsharma356
- **LinkedIn:** https://www.linkedin.com/in/harsh-sharma-a97819328/




# Results

## Task 01: Object Detection

### Approach
- Sourced the NuImages 5k and US Traffic Signs datasets from Roboflow Universe.
- Merged the datasets and created a custom dataset tailored to the requirements of this assignment.
- Selected **RF-DETR Nano**, a lightweight object detection model based on the Transformer architecture (Vision Transformer-based detection approach).
- Trained the model using transfer learning on the custom dataset to detect the following classes:
  - Cone
  - Barrier
  - Stop Sign

### Dataset Details

| Parameter | Value |
|---|---|
| Source | Roboflow Universe |
| Datasets Used | NuImages 5k, US Traffic Signs |
| Dataset Type | Custom merged dataset |
| Training Images | 3,407 |
| Validation Images | 652 |
| Test Images | 446 |
| Total Images | 4,505 |
| Total Annotations | 14,832 |
| Classes | Cone, Barrier, Stop Sign |

### Model Performance

| Metric | Result |
|---|---|
| Model | RF-DETR Nano |
| mAP@50 | 67.21% |
| mAP@50–95 | 45.21% |
| mAP@75 | 47.01% |
| Precision | 77.40% |
| Recall | 64.91% |
| F1-Score | 70.13% |
| mAR | 54.37% |
| Training Epochs | 300 |

### Conclusion
- Created a custom object detection dataset by merging NuImages 5k and US Traffic Signs datasets obtained from Roboflow Universe, adapting the data to the assignment's target classes.
- Trained RF-DETR Nano using transfer learning for detection of cones, barriers, and stop signs.
- The model achieved an mAP@50 of 67.21% and precision of 77.40%, demonstrating its ability to detect the target objects in the test data.
- The achieved recall of 64.91% indicates scope for further improvement in detecting all instances, particularly under challenging conditions such as occlusion, varying lighting, and small object sizes.

---

## Task 02: Distance Estimation

### Approach
- Calibrated the camera to determine its focal length in pixels.
- Used a cone as the reference object for calibration. The cone's physical height was measured from an image displayed on my phone, resulting in a relatively small reference height. This was sufficient for the initial calibration procedure.
- Applied the monocular distance estimation formula based on the pinhole camera model:

  $$Distance = \frac{Focal\ Length \times Actual\ Object\ Height}{Pixel\ Height}$$

- Calculated the focal length in pixels using the known reference distance, the cone's actual height, and its pixel height in the image.
- Incorporated the calibrated focal length into the distance estimation implementation.
- Used the average physical heights of the cone, barrier, and stop sign as reference dimensions for their respective distance calculations.
- Estimated the distance of each detected object and displayed the result in centimeters (cm), alongside the detected bounding box.

### Methodology

1. Captured an image of the reference cone at a known distance.
2. Measured the cone's physical height using the image displayed on a phone for calibration.
3. Obtained the cone's pixel height from its bounding box in the captured image.
4. Calculated the camera's focal length in pixels using the calibration formula.
5. Integrated the calculated focal length into the object detection and distance estimation pipeline.
6. Used class-specific average heights to estimate the distance of cones, barriers, and stop signs.

### Conclusion
- Calibrated the camera using a cone as a reference object to obtain the focal length in pixels.
- Implemented monocular distance estimation using the pinhole camera model.
- Incorporated class-specific average object heights to estimate the distances of cones, barriers, and stop signs.
- Displayed the estimated distance in centimeters alongside object detection results, integrating object detection and distance estimation into a unified pipeline.
- The accuracy of the distance estimation depends on camera calibration, bounding-box precision, the assumed physical height of each object, and the object's orientation relative to the camera.
---

## Task 03: Edge Optimization

### Approach

- Evaluated the trained RF-DETR Nano model on both CPU and GPU using the original PyTorch implementation to establish baseline performance.
- Exported the trained model to ONNX format for CPU inference and applied INT8 quantization to reduce model precision, computational overhead, and memory requirements.
- Used INT8 quantization for CPU deployment to reduce the computational cost of inference and potentially improve throughput on CPU hardware, where lower-precision integer operations can be more efficient than floating-point operations.
- Optimized GPU inference by exporting the model to TensorRT, using FP16 precision to leverage GPU-accelerated half-precision computation and TensorRT's engine-level optimizations, including graph optimization, kernel selection, and execution scheduling.
- Benchmarked each implementation using 20 warm-up iterations followed by 100 timed inference runs.
- Evaluated performance using average latency, median latency, and inference FPS.

### Benchmark Results

| Model / Backend | Device | Precision | Average Latency (ms) | Median Latency (ms) | Inference FPS |
|---|---|---|---:|---:|---:|
| RF-DETR (PyTorch) | CPU | FP32 | 96.29 | 95.63 | 10.39 |
| RF-DETR (ONNX Runtime) | CPU | INT8 | 77.87 | 78.79 | 12.84 |
| RF-DETR (PyTorch) | GPU | FP32 | 23.26 | 15.37 | 43.00 |
| RF-DETR (TensorRT) | GPU | FP16 | 7.52 | 4.69 | 133.02 |

### Performance Comparison

#### CPU Optimization: PyTorch vs ONNX INT8

| Metric | PyTorch CPU | ONNX INT8 CPU | Improvement |
|---|---:|---:|---:|
| Average Latency | 96.29 ms | 77.87 ms | 19.15% reduction |
| Median Latency | 95.63 ms | 78.79 ms | 17.61% reduction |
| Inference FPS | 10.39 | 12.84 | 23.58% increase |

#### GPU Optimization: PyTorch vs TensorRT FP16

| Metric | PyTorch GPU | TensorRT FP16 GPU | Improvement |
|---|---:|---:|---:|
| Average Latency | 23.26 ms | 7.52 ms | 67.67% reduction |
| Median Latency | 15.37 ms | 4.69 ms | 69.49% reduction |
| Inference FPS | 43.00 | 133.02 | 3.09× speedup |

### Conclusion

- INT8 quantization combined with ONNX Runtime improved CPU inference performance, reducing average latency from 96.29 ms to 77.87 ms and increasing inference throughput from 10.39 FPS to 12.84 FPS.
- INT8 quantization reduces the numerical precision and storage requirements of model weights and, where supported, intermediate computations. This can reduce computational cost and improve CPU inference efficiency.
- TensorRT FP16 optimization significantly improved GPU inference performance, reducing average latency by 67.67% compared with the measured PyTorch GPU baseline.
- The TensorRT engine achieved 7.52 ms average latency and 133.02 inference FPS, demonstrating the effectiveness of reduced-precision inference and hardware-specific engine optimization for GPU deployment.
- ONNX INT8 provided a measurable improvement for CPU-based deployment, while TensorRT FP16 delivered the highest inference throughput among the evaluated configurations.
- These optimizations demonstrate the potential of deploying the trained object detection model on resource-constrained and edge-computing platforms while improving inference speed.

---

## Overall Conclusion

- **Detection:** [Summary of detection performance.]
- **Distance Estimation:** [Summary of distance estimation performance.]
- **Edge Deployment:** [Summary of optimization results and selected deployment backend.]
- **Final Deployment Recommendation:** [Recommended model/backend for real-time deployment.]

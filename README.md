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

- **Method Used:** Monocular distance estimation using the pinhole camera model
- **Formula:** `Distance = (Focal Length × Actual Object Height) / Pixel Height`

### Results

| Object Class | Actual Height | Estimated Distance | Ground Truth Distance | Error |
|---|---|---|---|---|
| Cone | XX cm | XX cm | XX cm | XX cm |
| Barrier | XX cm | XX cm | XX cm | XX cm |
| Stop Sign | XX cm | XX cm | XX cm | XX cm |

### Observations
- [Mention the accuracy of distance estimation.]
- [Mention the effect of object size, bounding box accuracy, and camera calibration.]

---

## Task 03: Edge Optimization

### Benchmark Results

| Model / Backend | Precision | Device | Avg. Latency (ms) | Median Latency (ms) | Inference FPS |
|---|---|---|---|---|---|
| PyTorch | FP32 | CPU | XX | XX | XX |
| ONNX Runtime | FP32 | CPU | XX | XX | XX |
| PyTorch | FP32/FP16 | GPU | XX | XX | XX |
| TensorRT | FP16 | GPU | XX | XX | XX |
| ONNX Runtime | INT8 | CPU | XX | XX | XX |

### Optimization Comparison

- **Best Performing Backend:** [Backend]
- **Speedup Compared to Baseline:** XX×
- **Latency Reduction:** XX%
- **FPS Improvement:** XX%

### Observations
- [Compare CPU and GPU performance.]
- [Discuss the effect of quantization and inference optimization.]
- [Mention accuracy-performance trade-offs, if any.]
- [Highlight the most suitable deployment configuration.]

---

## Overall Conclusion

- **Detection:** [Summary of detection performance.]
- **Distance Estimation:** [Summary of distance estimation performance.]
- **Edge Deployment:** [Summary of optimization results and selected deployment backend.]
- **Final Deployment Recommendation:** [Recommended model/backend for real-time deployment.]

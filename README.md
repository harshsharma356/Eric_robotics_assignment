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

- **Model Used:** RF-DETR Nano
- **Dataset:** Merged NuImages 5k and US Traffic Signs dataset
- **Classes:** Cone, Barrier, Stop Sign

### Model Performance

| Metric | Result |
|---|---|
| mAP@50 | XX% |
| mAP@50–95 | XX% |
| mAP@75 | XX% |
| Precision | XX% |
| Recall | XX% |
| F1-Score | XX% |

### Observations
- [Add observations regarding detection accuracy, performance, and limitations.]
- [Mention any challenges encountered during training or inference.]

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

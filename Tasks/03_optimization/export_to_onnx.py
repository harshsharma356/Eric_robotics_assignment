from rfdetr import RFDETR

checkpoint = "/home/galdrux/internship_projs/eric_robotics/technical_assignnment_ericRobotics/model_training/checkpoint_best_total.pth"

model = RFDETR.from_checkpoint(checkpoint)

onnx_path = model.export(
    format="onnx",
    quantization="int8",
    calibration_data="/home/galdrux/internship_projs/eric_robotics/technical_assignnment_ericRobotics/Dataset/merged_dataset/train/images",
    max_images=100
)

print("INT8 ONNX exported:", onnx_path)
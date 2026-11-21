from rfdetr import RFDETR

checkpoint = "/home/galdrux/internship_projs/eric_robotics/technical_assignnment_ericRobotics/model_training/checkpoint_best_total.pth"
model = RFDETR.from_checkpoint(checkpoint)

model.export(format = "tensorrt")
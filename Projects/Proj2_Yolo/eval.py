import os
from ultralytics import YOLO

# Load the pre-trained model (after training is complete)
model = YOLO('Buoy_Detection_Results/YOLOv11n_buoy/weights/best.pt')  # Path to the best model from training

# Evaluate the model on the test dataset
metrics = model.val(data='Buoy-Detection-1/data.yaml',  # Path to the dataset YAML file
                    imgsz=640,                       # Image size for evaluation (same as training size)
                    conf=0.001,                      # Confidence threshold (you can adjust this)
                    iou=0.5,                         # Intersection over Union threshold
                    split="test",                    # Test dataset for evaluation
                    save_txt=True,                   # Optionally save the predictions to text files
                    save_json=True,)                  # Optionally save predictions to JSON for later analysis

# Access different mAP metrics
map50_95 = metrics.box.map  # mAP@0.5:0.95
map50 = metrics.box.map50    # mAP@0.5
map75 = metrics.box.map75    # mAP@0.75
category_map = metrics.box.maps  # mAP for each category (class) in the dataset

# Print the metrics
print(f"mAP@0.5:0.95: {map50_95:.4f}")
print(f"mAP@0.5: {map50:.4f}")
print(f"mAP@0.75: {map75:.4f}")
print("mAP for each category (class):")
for i, class_map in enumerate(category_map):
    print(f"Class {i}: {class_map:.4f}")

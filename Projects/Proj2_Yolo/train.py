import os
from ultralytics import YOLO

# Ensure CUDA blocking for debugging (if necessary)
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# TensorBoard logging directory
log_dir = 'runs/YOLO_Buoy_Detection'

# Load the pre-trained YOLO model (nano model for lightweight training)
model = YOLO('yolo11n.pt')  # Note: Ensure this model exists in your directory

# Training the model on the custom buoy detection dataset
# with additional configuration for TensorBoard and metric tracking
model.train(
    data='Buoy-Detection-1/data.yaml',  # Path to the dataset YAML file
    epochs=20,                            # Number of training epochs
    save=True,                           # Save model after training
    project='Buoy_Detection_Results',     # Folder to save results
    name='YOLOv11n_buoy',                 # Run name
    exist_ok=True,                        # Overwrite if exists
    verbose=True,                          # Detailed output
    batch=4
)
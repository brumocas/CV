# Import the dataset
from roboflow import Roboflow
rf = Roboflow(api_key="dnZcw1fNasJT5SaFbDdG")
project = rf.workspace("vortexbuoytrainingset").project("buoy-detection-qzjg1")
version = project.version(1)
dataset = version.download("yolov11")

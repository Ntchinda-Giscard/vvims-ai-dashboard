from roboflow import Roboflow
import os
rf = Roboflow(api_key="P4usj8uPwcbnflvyJIAB")
project = rf.workspace("ntchindagiscard").project("id-card-information")
version = project.version(1)
dataset = version.download("yolov8")


class DataIngestion:

    def __init__(self):
        roboflow_api_key = os.getenv("ROBOFLOW_API_KEY")

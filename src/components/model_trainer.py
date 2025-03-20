from abc import ABC
from typing import Type
from ultralytics import YOLO
from src.base import ModelTrainer
from src.entity.config_entity import ModelTrainerConfig
import os

class YoloModelTrainer(ModelTrainer):
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self, dataset) -> YOLO:
        # Get the CORRECT dataset path (fix nested folder issue)
        dataset_path = os.path.join(dataset.location, dataset.name)  # Fix nested folder
        data_yaml_path = os.path.join(dataset_path, "data.yaml")  # Correct YAML path

        # Verify paths (debugging)
        print(f"[DEBUG] Dataset root: {dataset.location}")
        print(f"[DEBUG] Expected dataset folder: {dataset_path}")
        print(f"[DEBUG] Contents of dataset.location: {os.listdir(dataset.location)}")
        print(f"[DEBUG] data.yaml exists: {os.path.exists(data_yaml_path)}")

        # Validate dataset structure
        assert os.path.exists(data_yaml_path), f"data.yaml not found at {data_yaml_path}"
        assert os.path.exists(os.path.join(dataset_path, "train", "images")), "Train images missing!"

        # Initialize model and train
        model = YOLO(self.config.model)
        results = model.train(
            data=data_yaml_path,  # Use corrected path
            epochs=self.config.epochs,
            batch=self.config.batch,
            imgsz=self.config.imgsz,
            device=self.config.device,
            scale=self.config.scale,
            mixup=self.config.mixup
        )
        return model
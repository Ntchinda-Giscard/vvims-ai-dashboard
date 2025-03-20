from abc import ABC
from typing import Type
from ultralytics import YOLO
from src.base import ModelTrainer
from pathlib import Path
from src.entity.config_entity import ModelTrainerConfig
import os


class YoloModelTrainer(ModelTrainer):
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self, dataset) -> YOLO:
        # Use dataset.location directly (no subfolder needed)
        data_yaml_path = os.path.join(dataset.location, "data.yaml")  # ✅ Correct path

        # Debugging
        print(f"[DEBUG] Dataset root: {dataset.location}")
        print(f"[DEBUG] Contents of dataset.location: {os.listdir(dataset.location)}")
        print(f"[DEBUG] data.yaml exists: {os.path.exists(data_yaml_path)}")

        # Validate paths
        assert os.path.exists(data_yaml_path), f"data.yaml not found at {data_yaml_path}"
        assert os.path.exists(os.path.join(dataset.location, "train", "images")), "Train images missing!"

        # Train
        model = YOLO(self.config.model)
        results = model.train(
            data=Path(dataset.location) / "data.yaml",  # Direct path to data.yaml
            epochs=self.config.epochs,
            batch=self.config.batch,
            imgsz=self.config.imgsz,
            device=self.config.device,
            scale=self.config.scale,
            mixup=self.config.mixup
        )
        return model
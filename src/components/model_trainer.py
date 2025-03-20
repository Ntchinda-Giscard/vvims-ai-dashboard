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
        model = YOLO(self.config.model)
        # Print dataset location
        print(f"Dataset location: {dataset.location}")

        # Verify folders exist
        print(os.listdir(dataset.location))  # Should show ["train", "valid", "data.yaml"]
        print(os.listdir(f"{dataset.location}/train"))  # Should show ["images", "labels"]

        results = model.train(
            data=f"{dataset.location}/data.yaml",
            epochs=self.config.epochs,
            batch=self.config.batch,
            imgsz=self.config.imgsz,
            device=self.config.device,
            scale=self.config.scale,
            mixup=self.config.mixup
        )

        return model

from abc import ABC
from typing import Type
from ultralytics import YOLO

from src.base import ModelTrainer
from src.entity.config_entity import ModelTrainerConfig


class YoloModelTrainer(ModelTrainer):

    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self) -> YOLO:
        model = self.config.model
        results = model.train(
            data=self.config.data,
            epochs=self.config.epochs,
            batch=self.config.batch,
            imgsz=self.config.imgsz,
            device=self.config.device,
            scale=self.config.scale,
            mixup=self.config.mixup
        )

        return model

from __future__ import annotations
from zenml.logger import get_logger
import json
import os
import typing as t
from pathlib import Path
import bentoml
from bentoml.validators import ContentType
import os

Image = t.Annotated[Path, ContentType("image/*")]
logger = get_logger(__name__)


@bentoml.service(resources={"gpu": 1})
class YoloV8:
    def __init__(self):
        from ultralytics import YOLO

        yolo_model = os.getenv("YOLO_MODEL", "yolov8x.pt")

        self.model = YOLO(Path("runs/detect/train6/weights/best.pt"))


    @bentoml.api
    def predict(self, image: Image):
        result = self.model.predict(image)[0]
        logger.info(f"Result : {result}")
        return {"result" : "Everything ok"}

# 

print(len(os.listdir("dataset")))
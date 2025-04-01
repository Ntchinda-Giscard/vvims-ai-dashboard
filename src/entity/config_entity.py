from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    workspace: str
    project: str
    version: int
    dataset: str
    roboflow_api_key: str

@dataclass
class ModelTrainerConfig:
    data: str
    model: str
    epochs: str
    batch: int
    imgsz: int
    device: str
    scale: int
    mixup: int

@dataclass
class MilDatasetConfig:
    anomal_video_dir: str
    normal_video_dir: str
    classes: dict
    
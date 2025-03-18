from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    workspace: str
    project: str
    version: int
    dataset: str
    roboflow_api_key: str
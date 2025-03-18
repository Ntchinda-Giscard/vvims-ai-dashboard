from roboflow import Roboflow
from src.base import DataIngestor
from src.entity.config_entity import DataIngestionConfig


class DataIngestion(DataIngestor):

    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def ingest_data(self):
        print(f">>>>>>>> Roboflow api key {self.config.roboflow_api_key}")
        rf = Roboflow(api_key=self.config.roboflow_api_key)
        project = rf.workspace(self.config.workspace).project(self.config.project)
        version = project.version(self.config.version)

        dataset = version.download(self.config.dataset)

        return  dataset

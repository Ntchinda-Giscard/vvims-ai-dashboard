from roboflow import Roboflow
from src.base import DataIngestor
from src.config.get_config import ConfigManager


class DataIngestion(DataIngestor):

    def __init__(self, config: ConfigManager):
        self.config = config.get_data_ingestion_config()

    def ingest_data(self):
        rf = Roboflow(api_key=self.config.roboflow_api_key)
        project = rf.workspace(self.config.workspace).project(self.config.project)
        version = project.version(self.config.version)

        dataset = version.download(self.config.dataset)

        return  dataset

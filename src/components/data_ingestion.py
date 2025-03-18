from roboflow import Roboflow
from src import DataIngestionConfig
from src.base import DataIngestor




class DataIngestion(DataIngestor):

    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def ingest_data(self):
        rf = Roboflow(api_key=self.config.roboflow_api_key)
        project = rf.workspace(self.config.workspace).project(self.config.project)
        version = project.version(self.config.version)

        dataset = version.download(self.config.dataset)

        return  dataset

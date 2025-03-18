
from src import read_yaml
import os

from src.constants import CONFIG_FILE_PATH
from src.entity.config_entity import DataIngestionConfig


class ConfigManager:

    def __init__(self, config_filepath = CONFIG_FILE_PATH):
        self.config = read_yaml(config_filepath)


    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        api_key = os.getenv("ROBOFLOW_API_KEY")
        data_ingestion_config = DataIngestionConfig(
            workspace=config.workspace,
            project=config.project,
            version=config.version,
            dataset=config.dataset,
            roboflow_api_key="P4usj8uPwcbnflvyJIAB"
        )
        print(f">>>>>>>>>>>>> Configurations {data_ingestion_config}")

        return data_ingestion_config
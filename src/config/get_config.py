from src import CONFIG_FILE_PATH
from src import read_yaml

class ConfigManager:

    def __init__(self, config_filepath = CONFIG_FILE_PATH):
        self.config = read_yaml(CONFIG_FILE_PATH)


    def get_data_ingestion_config(self):

        pass
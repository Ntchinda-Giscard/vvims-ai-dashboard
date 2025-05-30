from typing import ( Annotated)
import roboflow
from zenml import step
from src.components.data_ingestion import DataIngestion
from src.config.get_config import ConfigManager
from zenml.logger import get_logger


logger = get_logger(__name__)
STAGE_NAME = "Data Ingestion"


class DataIngestionStep:

    def __init__(self) -> None:

        self.config = ConfigManager()

    def main(self):
        data_loader_config = self.config.get_data_ingestion_config()
        data_ingestor = DataIngestion(data_loader_config)
        dataset = data_ingestor.ingest_data()
        return dataset



@step(enable_cache=False)
def data_ingestion() -> Annotated[roboflow.core.dataset.Dataset, "dataset"]:
    try:
        logger.info(f"\33[33m>>>>> 1️⃣ {STAGE_NAME}📀 step has started 🏁🏁 <<<<<\33[0m")
        obj = DataIngestionStep()
        dataset = obj.main()
        logger.info(f"\33[33m>>>>> ✅ {STAGE_NAME}📀 step has completed x=========x\33[0m")
        return dataset
    except Exception as e:
        logger.exception(f"Oops😟! An error occurred: {e} ")


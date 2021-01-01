from ultralytics import YOLO
from zenml.logger import get_logger
from zenml import step
from src.components.model_trainer import YoloModelTrainer
from src.config.get_config import ConfigManager
from typing import Annotated
from src.materializers.yolo_materializer import UltralyticsMaterializer
from roboflow.core.dataset import Dataset
logger = get_logger(__name__)

STAGE_NAME = "Model Trainer"


class ModelTrainerStep:

    def __init__(self):
        self.config = ConfigManager()

    def main(self, dataset):

        model_trainer_config = self.config.get_model_trainer()
        model_trainer = YoloModelTrainer(model_trainer_config)
        model = model_trainer.train(dataset)

        return model

@step(output_materializers=UltralyticsMaterializer, enable_cache=False)
def model_trainer_step(dataset: Dataset) -> Annotated[ YOLO ,"model"]:
    try:
        logger.info(f"\33[33m>>>>> 1️⃣ {STAGE_NAME}📀 step has started 🏁🏁 <<<<<\33[0m")
        obj = ModelTrainerStep()
        model = obj.main(dataset)
        logger.info(f"\33[33m>>>>> ✅ {STAGE_NAME}📀 step has completed x=========x\33[0m")
        return model
    except Exception as e:
        logger.exception(f"Oops😟! An error occurred: {e} ")

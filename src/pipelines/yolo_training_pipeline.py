from src import data_ingestion
from zenml import pipeline

from src.steps.step2_model_trainer import model_trainer_step


@pipeline
def training_pipeline():
    dataset = data_ingestion()
    model = model_trainer_step(dataset)
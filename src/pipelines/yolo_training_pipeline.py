from src import data_ingestion
from zenml import pipeline
from src.steps.step2_model_trainer import model_trainer_step
from src.steps.step3_model_promotion import promote_to_bentoml


@pipeline
def training_pipeline():
    dataset = data_ingestion()
    model = model_trainer_step(dataset)
    promote_to_bentoml(model)
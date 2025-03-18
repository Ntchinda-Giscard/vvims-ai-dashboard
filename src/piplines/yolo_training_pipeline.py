from src import data_ingestion
from zenml import pipeline

@pipeline
def training_pipeline():
    data_ingestion()
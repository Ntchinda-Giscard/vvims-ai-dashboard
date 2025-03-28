from zenml import step, get_step_context
from zenml.client import Client
from ultralytics import YOLO
import bentoml
from bentoml._internal.models.model import Model

import numpy as np

@step(enable_cache=False)
def promote_to_bentoml(
    model: YOLO) -> None:  # Explicit return type

    bentoml_model: Model = bentoml.pytorch.save_model("ID_CARD_DETECTOR", model)
    print(f"Tag oooo: {bentoml_model.tag}")
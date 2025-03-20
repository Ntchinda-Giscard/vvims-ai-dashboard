import os
import yaml  # or ruamel.yaml for better YAML handling
from ultralytics import YOLO
from src.base import ModelTrainer
from src.entity.config_entity import ModelTrainerConfig

class YoloModelTrainer(ModelTrainer):
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self, dataset) -> YOLO:
        # Path to the original YAML file
        data_yaml_path = os.path.join(dataset.location, "data.yaml")

        # Load the YAML content into a dictionary
        with open(data_yaml_path, "r") as f:
            data_config = yaml.safe_load(f)

        # Override the paths to remove the nested folder
        data_config["train"] = os.path.join(dataset.location, "train", "images")
        data_config["val"] = os.path.join(dataset.location, "valid", "images")
        data_config["test"] = os.path.join(dataset.location, "test", "images")  # optional

        # Debugging: Print corrected paths
        print(f"[DEBUG] Corrected train path: {data_config['train']}")
        print(f"[DEBUG] Corrected val path: {data_config['val']}")

        # Add this to your code
        assert os.path.exists(data_config["train"]), f"Train images missing: {data_config['train']}"
        assert os.path.exists(data_config["val"]), f"Validation images missing: {data_config['val']}"

        # Initialize model and train with the corrected config
        model = YOLO(self.config.model)
        results = model.train(
            data=data_config,  # Pass the corrected config dictionary (not the YAML file path)
            epochs=self.config.epochs,
            batch=self.config.batch,
            imgsz=self.config.imgsz,
            device=self.config.device,
            scale=self.config.scale,
            mixup=self.config.mixup
        )
        return model
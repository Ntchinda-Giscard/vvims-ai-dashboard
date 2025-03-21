import os
import yaml
import tempfile
from ultralytics import YOLO
from src.base import ModelTrainer
from src.entity.config_entity import ModelTrainerConfig

class YoloModelTrainer(ModelTrainer):
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self, dataset) -> YOLO:
        # Load the original YAML content
        data_yaml_path = os.path.join(dataset.location, "data.yaml")
        with open(data_yaml_path, "r") as f:
            data_config = yaml.safe_load(f)

        # Correct the paths (remove nested folders)
        data_config["train"] = os.path.join(dataset.location, "train", "images")
        data_config["val"] = os.path.join(dataset.location, "valid", "images")
        data_config["test"] = os.path.join(dataset.location, "test", "images")  # optional

        # Create a temporary YAML file with corrected paths
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".yaml") as tmp_file:
            yaml.dump(data_config, tmp_file)
            temp_yaml_path = tmp_file.name

        # Debugging: Print paths
        print(f"[DEBUG] Temporary YAML path: {temp_yaml_path}")
        print(f"[DEBUG] Train path: {data_config['train']}")
        
        assert os.path.exists(data_config["train"]), f"Missing train images: {data_config['train']}"
        assert os.path.exists(data_config["val"]), f"Missing validation images: {data_config['val']}"

        # Train with the temporary YAML file
        model = YOLO(self.config.model)
        results = model.train(
            data=temp_yaml_path,  # Pass the temporary YAML path (string)
            epochs=self.config.epochs,
            batch=self.config.batch,
            imgsz=self.config.imgsz,
            device=self.config.device,
            scale=self.config.scale,
            mixup=self.config.mixup
        )

        # Clean up the temporary file (optional)
        os.remove(temp_yaml_path)

        return model
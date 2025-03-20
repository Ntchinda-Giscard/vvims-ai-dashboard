import json
import os
from typing import Type
from zenml.materializers.base_materializer import BaseMaterializer
from zenml.enums import ArtifactType
import roboflow
import supervision as sv


class RoboflowDatasetMaterializer(BaseMaterializer):
    ASSOCIATED_TYPES = (roboflow.core.dataset.Dataset,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.DATA

    def load(self, data_type: Type[roboflow.core.dataset.Dataset]) -> roboflow.core.dataset.Dataset:
        """Load the dataset from the artifact store."""
        # with self.artifact_store.open(os.path.join(self.uri, 'metadata.json'), 'r') as f:
        #     metadata = json.load(f)
        # # Use available methods to reconstruct the dataset
        # dataset = roboflow.core.dataset.Dataset.from_metadata(metadata)
        dataset = sv.DetectionDataset.from_yolo(
            images_directory_path=f"{data_type.location}/train/images",
            annotations_directory_path=f"{data_type.location}/train/labels",
            data_yaml_path=f"{data_type.location}/data.yaml"
        )
        return dataset

    def save(self, dataset: roboflow.core.dataset.Dataset) -> None:
        """Save dataset metadata to the artifact store."""

        metadata = {
            "name": dataset.name if hasattr(dataset, "name") else "Unknown",
            "version": dataset.version if hasattr(dataset, "version") else "Unknown",
            "size": dataset.size if hasattr(dataset, "size") else "Unknown",
            "classes": dataset.classes if hasattr(dataset, "classes") else [],
        }

        with self.artifact_store.open(os.path.join(self.uri, "metadata.json"), "w") as f:
            json.dump(metadata, f)


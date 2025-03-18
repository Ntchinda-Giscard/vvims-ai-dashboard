import json
import os
from typing import Type
from zenml.materializers.base_materializer import BaseMaterializer
from zenml.enums import ArtifactType
import roboflow

class RoboflowDatasetMaterializer(BaseMaterializer):
    ASSOCIATED_TYPES = (roboflow.core.dataset.Dataset,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.DATA

    def load(self, data_type: Type[roboflow.core.dataset.Dataset]) -> roboflow.core.dataset.Dataset:
        """Load the dataset from the artifact store."""
        with self.artifact_store.open(os.path.join(self.uri, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
        # Modify this if roboflow has a specific method for loading datasets from metadata
        dataset = roboflow.Roboflow().workspace().project(metadata["name"]).version(metadata["version"]).download()
        return dataset

    def save(self, dataset: roboflow.core.dataset.Dataset) -> None:
        """Save dataset metadata to the artifact store."""
        metadata = {
            "name": dataset.name,
            "version": dataset.version,
            "num_images": len(dataset.images),
            "classes": dataset.classes if hasattr(dataset, "classes") else None,
        }
        with self.artifact_store.open(os.path.join(self.uri, "metadata.json"), "w") as f:
            json.dump(metadata, f)

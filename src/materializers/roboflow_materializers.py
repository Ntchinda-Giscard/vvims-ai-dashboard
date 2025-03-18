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
        # Example: Load dataset metadata from a JSON file
        with self.artifact_store.open(os.path.join(self.uri, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
        # Use metadata to reconstruct the dataset
        dataset = roboflow.core.dataset.Dataset.from_metadata(metadata)
        return dataset

    def save(self, dataset: roboflow.core.dataset.Dataset) -> None:
        """Save the dataset to the artifact store."""
        # Example: Save dataset metadata to a JSON file
        metadata = dataset.to_metadata()
        with self.artifact_store.open(os.path.join(self.uri, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)

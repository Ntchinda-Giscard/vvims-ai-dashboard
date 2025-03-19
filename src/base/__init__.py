from abc import ABC, abstractmethod
from typing import Tuple, Type


class DataIngestor(ABC):
    @abstractmethod
    def ingest_data(self) -> Type[NotImplementedError]:
        return NotImplementedError

class ModelTrainer(ABC):
    @abstractmethod
    def train(self) -> Type[NotImplementedError]:
        return NotImplementedError
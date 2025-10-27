from abc import ABC, abstractmethod
from core.Configs import DataConfig

class BaseUtilityModel(ABC):
    def __init__(self, config: DataConfig, name="base_utility_model"):
        self.name = name
        self.config = config

    def get_name(self):
        return self.name

from abc import ABC, abstractmethod


class Component(ABC):
    @classmethod
    @abstractmethod
    def param_func(cls):
        pass

    @classmethod
    @abstractmethod
    def model_func(cls):
        pass

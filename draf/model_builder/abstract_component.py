from abc import ABC, abstractmethod


class Component(ABC):
    @abstractmethod
    def param_func(self, sc):
        pass

    @abstractmethod
    def model_func(self, sc, m, d, p, v):
        pass

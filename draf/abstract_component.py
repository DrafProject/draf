from abc import ABC, abstractmethod


class Component(ABC):
    def dim_func(self, sc) -> None:
        pass

    @abstractmethod
    def param_func(self, sc) -> None:
        pass

    @abstractmethod
    def model_func(self, sc, m, d, p, v, c) -> None:
        pass

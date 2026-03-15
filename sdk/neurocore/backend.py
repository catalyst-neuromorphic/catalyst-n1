from abc import ABC, abstractmethod

class Backend(ABC):

    @abstractmethod
    def deploy(self, network_or_compiled): ...

    @abstractmethod
    def inject(self, target, current): ...

    @abstractmethod
    def run(self, timesteps): ...

    @abstractmethod
    def set_learning(self, learn=False, graded=False, dendritic=False,
                     async_mode=False, three_factor=False, noise=False): ...

    @abstractmethod
    def reward(self, value): ...

    @abstractmethod
    def status(self): ...

    @abstractmethod
    def close(self): ...

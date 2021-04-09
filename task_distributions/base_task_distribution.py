import abc


class BaseTaskDistribution(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def sample(self, num_tasks: int):
        pass

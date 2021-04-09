import abc


class BaseTask(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def sample_data(self, num_datapoints: int):
        pass

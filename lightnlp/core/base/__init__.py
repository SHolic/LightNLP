from abc import abstractmethod

from lightnlp.utils.visualizer import SummaryWriter, ModelVisualizer


class BaseModelMixin:
    def __init__(self):
        self.summary_writer = SummaryWriter()
        self.model_visualizer = ModelVisualizer()
        self.model = None

    @abstractmethod
    def train(self, *args, **kwargs):
        pass

    @abstractmethod
    def predict(self, *args, **kwargs):
        pass

    @abstractmethod
    def save(self, *args, **kwargs):
        pass

    @staticmethod
    @abstractmethod
    def load(*args, **kwargs):
        pass

    def visualize(self, name="loss"):
        # loss, lr
        if name == "model":
            if self.model is None:
                raise NotImplementedError("It need to train first!")
            self.model_visualizer.visualize(model=self.model)
        else:
            self.summary_writer.plot(title=name)

from abc import abstractmethod
import torch

from lightnlp.utils.visualizer import SummaryWriter, ModelVisualizer


class BaseModelMixin:
    def __init__(self, device):
        self.summary_writer = SummaryWriter()
        self.model_visualizer = ModelVisualizer()
        self.model = None
        self.device = torch.device(device) if isinstance(device, str) else None
        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    @abstractmethod
    def train(self, *args, **kwargs):
        pass

    @abstractmethod
    def predict(self, *args, **kwargs):
        pass

    def save(self, path):
        torch.save({
            'ins': self
        }, path)
        return self

    @staticmethod
    def load(path):
        return torch.load(path)['ins']

    def visualize(self, name="loss"):
        # loss, lr
        if name == "model":
            if self.model is None:
                raise NotImplementedError("It need to train first!")
            self.model_visualizer.visualize(model=self.model)
        else:
            self.summary_writer.plot(title=name)

from abc import abstractmethod
import torch

from lightnlp.utils.visualizer import SummaryWriter, ModelVisualizer
from ...utils.data_utils import RawDataLoader


class BaseModelMixin:
    def __init__(self, device):
        self.summary_writer = SummaryWriter()
        self.model_visualizer = ModelVisualizer()
        self.model = None
        self.device = torch.device(device) if isinstance(device, str) else None
        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.data_loader = None
        self.trainer = None

    @abstractmethod
    def train(self, *args, **kwargs):
        pass

    def predict(self, data=None, data_path=None, batch_size=64, verbose=1, n_jobs=1, **kwargs):
        if self.model is None or self.data_loader is None \
                or self.trainer is None or self.summary_writer is None:
            raise NotImplementedError("Predict function should be executed after train function!")

        self.data_loader.verbose = verbose
        corpus = list()
        if data is not None:
            corpus += [data] if isinstance(data, str) else data
        if data_path is not None:
            corpus += RawDataLoader(verbose=verbose).load_predict(data_path)

        test_data = self.data_loader.load_predict(corpus, batch_size=batch_size, n_jobs=n_jobs)

        return self.trainer.predict(model=self.model, test_data=test_data,
                                    summary_writer=self.summary_writer, verbose=verbose,
                                    label2idx=self.data_loader.label2idx)

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

from pathlib import PurePath
import torch
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup

from ..base import BaseModelMixin
from ...utils.data_utils import RawDataLoader, AlbertBaseATCDataLoader
from ...utils.visualizer import SummaryWriter
from ...nn.models.atc import AlbertClassificationModel
from ...trainer import ATCTrainer


default_albert_model_path = PurePath(__file__).parent.parent.parent / "res/default_albert_model/"


class AlbertClassification(BaseModelMixin):
    def __init__(self,
                 pre_trained_model_path=default_albert_model_path,
                 device=None,
                 hidden_dim=100,
                 lr=1e-5,
                 eps=1e-8,
                 train_size=0.9,
                 batch_size=64,
                 epoch_num=5,
                 dropout_rate=0.3,
                 random_state=2020,
                 linear_activation=nn.ReLU,
                 embed_finetune=True,
                 n_jobs=1,
                 verbose=1
                 ):
        super(AlbertClassification, self).__init__()
        self.pre_trained_model_path = pre_trained_model_path
        self.device = device
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.eps = eps
        self.train_size = train_size
        self.batch_size = batch_size
        self.epoch_num = epoch_num
        self.dropout_rate = dropout_rate
        self.random_state = random_state
        self.linear_activation = linear_activation
        self.embed_finetune = embed_finetune
        self.n_jobs = n_jobs
        self.verbose = verbose

        self.model = None
        self.data_loader = None
        self.trainer = None
        self.summary_writer = None

    def train(self, data_path=None, corpus=None, label=None):
        # prepare data for Trainer
        if data_path is not None:
            corpus, label = RawDataLoader(verbose=self.verbose).load_train(path=data_path, file_use="atc")

        data_loader = AlbertBaseATCDataLoader(pre_trained_path=self.pre_trained_model_path,
                                              max_length=None,
                                              batch_size=self.batch_size,
                                              train_size=self.train_size,
                                              random_state=self.random_state,
                                              verbose=self.verbose)
        train_data, dev_data = data_loader.load_train(corpus=corpus, label=label, n_jobs=self.n_jobs)

        model = AlbertClassificationModel(label_num=data_loader.label_num,
                                          hidden_dim=self.hidden_dim,
                                          finetune=self.embed_finetune,
                                          pre_trained_model_path=self.pre_trained_model_path,
                                          linear_activation=self.linear_activation,
                                          dropout_rate=self.dropout_rate)

        optimizer = AdamW(model.parameters(), lr=self.lr, eps=self.eps)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=int(len(train_data) * self.epoch_num * 0.05),
                                                    num_training_steps=len(train_data) * self.epoch_num)
        summary_writer = SummaryWriter()

        trainer = ATCTrainer(device=self.device, batch_size=self.batch_size, epoch_num=self.epoch_num)
        trainer.train(model=model, train_data=train_data, dev_data=dev_data,
                      optimizer=optimizer, scheduler=scheduler, summary_writer=summary_writer,
                      verbose=self.verbose, label2idx=data_loader.label2idx)

        # summary_writer.plot(title="loss")

        self.model = model
        self.data_loader = data_loader
        self.trainer = trainer
        self.summary_writer = summary_writer
        return self

    def predict(self, data=None, data_path=None, batch_size=64, verbose=1, n_jobs=1):
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
            'model': self.model,
            'data_loader': self.data_loader,
            'trainer': self.trainer,
            'summary_writer': self.summary_writer
        }, path)
        return self

    @staticmethod
    def load(path):
        cp = torch.load(path)
        ins = AlbertClassification()
        ins.model = cp["model"]
        ins.data_loader = cp["data_loader"]
        ins.trainer = cp["trainer"]
        ins.summary_writer = cp["summary_writer"]
        return ins


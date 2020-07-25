import torch
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup

from ..base import BaseModelMixin
from ...utils.data_utils import RawDataLoader, EmbeddingLoader, ATCDataLoader
from ...utils.visualizer import SummaryWriter
from ...nn.models.atc import CNNTextClassificationModel
from ...trainer import ATCTrainer


class CNNTextClassification(BaseModelMixin):
    def __init__(self,
                 device=None,
                 lr=1e-4,
                 eps=1e-8,
                 train_size=0.9,
                 batch_size=64,
                 epoch_num=5,
                 dropout_rate=0.3,
                 kernel_size=(3, 4, 5),
                 kernel_num=200,
                 random_state=2020,
                 linear_activation=nn.ReLU,
                 embed_finetune=True,
                 embed_sep=" ",
                 embed_dim=200,
                 embed_start="[SOS]",
                 embed_end="[END]",
                 embed_pad="[PAD]",
                 embed_unknown="[UNK]",
                 pre_trained_embed_path=None,
                 embed_words_path=None,
                 n_jobs=1,
                 verbose=1
                 ):
        super(CNNTextClassification, self).__init__(device=device)
        self.lr = lr
        self.eps = eps
        self.train_size = train_size
        self.batch_size = batch_size
        self.epoch_num = epoch_num
        self.dropout_rate = dropout_rate
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.random_state = random_state
        self.linear_activation = linear_activation
        self.embed_finetune = embed_finetune
        self.embed_sep = embed_sep
        self.embed_dim = embed_dim
        self.embed_start = embed_start
        self.embed_end = embed_end
        self.embed_pad = embed_pad
        self.embed_unknown = embed_unknown
        self.pre_trained_embed_path = pre_trained_embed_path
        self.embed_words_path = embed_words_path
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

        embed_loader = EmbeddingLoader(sep=self.embed_sep,
                                       dim=self.embed_dim,
                                       start=self.embed_start,
                                       end=self.embed_end,
                                       pad=self.embed_pad,
                                       unknown=self.embed_unknown,
                                       random_state=self.random_state,
                                       verbose=self.verbose)
        embed_weights, vocab2idx, idx2vocab = embed_loader.load(pre_trained_path=self.pre_trained_embed_path,
                                                                corpus=corpus,
                                                                word_path=self.embed_words_path)

        data_loader = ATCDataLoader(train_size=self.train_size,
                                    batch_size=self.batch_size,
                                    vocab2idx=vocab2idx,
                                    unknown_label=self.embed_unknown,
                                    pad_label=self.embed_pad,
                                    max_length=None,
                                    random_state=self.random_state,
                                    verbose=self.verbose)
        train_data, dev_data = data_loader.load_train(corpus=corpus, label=label, n_jobs=self.n_jobs)

        model = CNNTextClassificationModel(label_num=data_loader.label_num,
                                           vocab_size=embed_weights.size()[0],
                                           embedding_dim=embed_weights.size()[1],
                                           finetune=self.embed_finetune,
                                           pre_trained_embed_weights=embed_weights,
                                           kernel_size=self.kernel_size,
                                           kernel_num=self.kernel_num,
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
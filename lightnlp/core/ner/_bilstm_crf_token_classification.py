import torch
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup

from ...core.base import BaseModelMixin
from ...utils.data_utils import RawDataLoader, EmbeddingLoader, NERDataLoader
from ...utils.visualizer import SummaryWriter
from ...nn.models.ner import BiLSTMCRFTokenClassificationModel
from ...trainer import NERTrainer


class BiLSTMCRFClassification(BaseModelMixin):
    def __init__(self, ):
        super(BiLSTMCRFClassification, self).__init__()
        self.model = None
        self.data_loader = None
        self.trainer = None
        self.summary_writer = None

    def train(self,
              data_path,
              device=None,
              lr=1e-4,
              eps=1e-8,
              train_size=0.9,
              batch_size=64,
              epoch_num=5,
              dropout_rate=0.3,
              random_state=2020,
              linear_activation=nn.ReLU,
              embed_finetune=True,
              embed_sep=" ",
              embed_dim=200,
              hidden_dim=100,
              cell_type="lstm",
              embed_start="[SOS]",
              embed_end="[END]",
              embed_pad="[PAD]",
              embed_unknown="[UNK]",
              pre_trained_embed_path=None,
              embed_words_path=None,
              n_jobs=1,
              verbose=1,
              ):
        # prepare data for Trainer
        corpus, label = RawDataLoader(verbose=verbose).load_train(path=data_path, file_use="ner")

        embed_loader = EmbeddingLoader(sep=embed_sep, dim=embed_dim, start=embed_start,
                                       end=embed_end, pad=embed_pad, unknown=embed_unknown,
                                       random_state=random_state, verbose=verbose)
        embed_weights, vocab2idx, idx2vocab = embed_loader.load(pre_trained_path=pre_trained_embed_path,
                                                                corpus=corpus, word_path=embed_words_path)

        data_loader = NERDataLoader(train_size=train_size, batch_size=batch_size,
                                    vocab2idx=vocab2idx, unknown_label=embed_unknown,
                                    pad_label=embed_pad, max_length=None,
                                    random_state=random_state, verbose=verbose)

        train_data, dev_data = data_loader.load_train(corpus=corpus, label=label, n_jobs=n_jobs)

        model = BiLSTMCRFTokenClassificationModel(label_num=data_loader.label_num,
                                                  label2idx=data_loader.label2idx,
                                                  vocab_size=embed_weights.size()[0],
                                                  embedding_dim=embed_weights.size()[1],
                                                  hidden_dim=hidden_dim,
                                                  finetune=embed_finetune,
                                                  cell_type=cell_type,
                                                  pre_trained_embed_weights=embed_weights,
                                                  linear_activation=linear_activation,
                                                  dropout_rate=dropout_rate)

        optimizer = AdamW(model.parameters(), lr=lr, eps=eps)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=int(len(train_data) * epoch_num * 0.05),
                                                    num_training_steps=len(train_data) * epoch_num)
        summary_writer = SummaryWriter()

        trainer = NERTrainer(device=device, batch_size=batch_size, epoch_num=epoch_num)
        trainer.train(model=model, train_data=train_data, dev_data=dev_data,
                      optimizer=optimizer, scheduler=scheduler, summary_writer=summary_writer,
                      verbose=verbose, label2idx=data_loader.label2idx)

        self.model = model
        self.data_loader = data_loader
        self.trainer = trainer
        self.summary_writer = summary_writer
        return self

    # def predict(self, data=None, data_path=None, batch_size=64, verbose=1, n_jobs=1):
    #     if self.model is None or self.data_loader is None \
    #             or self.trainer is None or self.summary_writer is None:
    #         raise NotImplementedError("Predict function should be executed after train function!")
    #
    #     self.data_loader.verbose = verbose
    #     corpus = list()
    #     if data is not None:
    #         corpus += [data] if isinstance(data, str) else data
    #     if data_path is not None:
    #         corpus += RawDataLoader(verbose=verbose).load_predict(data_path)
    #
    #     test_data = self.data_loader.load_predict(corpus, batch_size=batch_size, n_jobs=n_jobs)
    #
    #     return self.trainer.predict(model=self.model, test_data=test_data,
    #                                 summary_writer=self.summary_writer, verbose=verbose,
    #                                 label2idx=self.data_loader.label2idx)

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
        ins = BiLSTMCRFClassification()
        ins.model = cp["model"]
        ins.atc_loader = cp["data_loader"]
        ins.trainer = cp["trainer"]
        ins.summary_writer = cp["summary_writer"]
        return ins

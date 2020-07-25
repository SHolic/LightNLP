import torch
import torch.nn as nn

from ...layers import EmbeddingLayer, RNNLayer, CrfLayer, LinearLayer


class BiLstmCrfTokenClassificationModel(nn.Module):
    def __init__(self,
                 label_num,
                 label2idx,
                 vocab_size,
                 embedding_dim,
                 hidden_dim,
                 finetune=True,
                 cell_type="lstm",
                 pre_trained_embed_weights=None,
                 linear_activation=None,
                 dropout_rate=0.3
                 ):

        super(BiLstmCrfTokenClassificationModel, self).__init__()
        self.label_num = label_num
        self.label2idx = label2idx
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.cell_type = cell_type
        self.finetune = finetune
        self.pre_trained_embed_weights = pre_trained_embed_weights

        self.linear_activation = linear_activation
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(self.dropout_rate)

        self.embedding = EmbeddingLayer(self.vocab_size, self.embedding_dim,
                                        self.pre_trained_embed_weights, self.finetune)

        self.lstm = RNNLayer(input_dim=self.embedding_dim, hidden_dim=self.hidden_dim//2,
                             num_layers=1, bidirectional=True, batch_first=False, cell_type=self.cell_type)

        self.linear = LinearLayer(input_dim=self.hidden_dim, output_dim=self.label_num,
                                  bias=True, activation=self.linear_activation)

        self.crf = CrfLayer(label2idx=self.label2idx)

    def forward(self, inputs=None, labels=None, only_loss=False, **kwargs):
        embs = self.embedding(inputs)
        feats = self.lstm(embs)
        emission = self.linear(feats)
        emission2 = self.dropout(emission)
        score, tag_seq, loss = self.crf(emission=emission2, label_ids=labels, only_loss=only_loss)

        return tag_seq, loss

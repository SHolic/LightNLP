import torch
import torch.nn as nn

from ...layers import EmbeddingLayer, RNNLayer, LinearLayer


class BiLSTMClassificationModel(nn.Module):
    def __init__(self,
                 label_num,
                 vocab_size,
                 embedding_dim,
                 finetune=True,
                 pre_trained_embed_weights=None,
                 hidden_dim=100,
                 num_layers=1,
                 bidirectional=True,
                 batch_first=True,
                 batch_size=1,
                 cell_type='lstm',
                 linear_activation=None,
                 dropout_rate=0.3,
                 device=None
                 ):
        super(BiLSTMClassificationModel, self).__init__()
        self.label_num = label_num
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.finetune = finetune
        self.pre_trained_embed_weights = pre_trained_embed_weights
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        self.batch_size = batch_size
        self.cell_type = cell_type
        self.linear_activation = linear_activation
        self.dropout_rate = dropout_rate
        self.device = device

        self.dropout = nn.Dropout(self.dropout_rate)
        self.softmax = torch.nn.Softmax(dim=1)
        self.loss = torch.nn.MSELoss()

        self.embedding = EmbeddingLayer(self.vocab_size, self.embedding_dim,
                                        self.pre_trained_embed_weights, self.finetune)
        self.lstm = RNNLayer(input_dim=self.embedding_dim,
                             hidden_dim=self.hidden_dim,
                             num_layers=self.num_layers,
                             bidirectional=self.bidirectional,
                             batch_first=self.batch_first,
                             batch_size=self.batch_size,
                             cell_type=self.cell_type,
                             device=self.device)
        self.linear = LinearLayer(self.hidden_dim, self.label_num, True, self.linear_activation)

    def forward(self, inputs=None, labels=None, mask=None, batch_size=None, **kwargs):
        if batch_size is None:
            batch_size = self.batch_size
        x1 = self.embedding(inputs)
        x2 = self.lstm(inputs=x1, mask=mask, return_type="one", batch_size=batch_size)
        x2 = self.dropout(x2)

        x3 = self.linear(x2)
        logits = self.softmax(x3)
        outputs = (logits,)

        if labels is not None:
            loss = self.loss(logits, labels)
            outputs = outputs + (loss,)
        else:
            outputs = outputs + (None,)
        return outputs  # logits, loss

import torch
import torch.nn as nn

from ...layers import EmbeddingLayer, CNNLayer, LinearLayer


class CNNTextClassificationModel(nn.Module):
    def __init__(self,
                 label_num,
                 vocab_size,
                 embedding_dim,
                 finetune=True,
                 pre_trained_embed_weights=None,
                 kernel_size=(3, 4, 5),
                 kernel_num=200,
                 linear_activation=None,
                 dropout_rate=0.3
                 ):
        """

        :param vocab_size:
        :param embedding_dim:
        :param pre_trained_embeddings:
        :param kernel_size: default (3,4,5)
        :param kernel_num:  default 200
        :param pool1d: default maxpool1d
        :param linear_activation: dufault ReLU
        """
        super(CNNTextClassificationModel, self).__init__()
        self.label_num = label_num
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.finetune = finetune
        self.pre_trained_embed_weights = pre_trained_embed_weights
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.linear_activation = linear_activation
        self.dropout_rate = dropout_rate

        self.dropout = nn.Dropout(self.dropout_rate)
        self.softmax = torch.nn.Softmax(dim=1)
        self.loss = torch.nn.MSELoss()

        self.embedding = EmbeddingLayer(self.vocab_size, self.embedding_dim,
                                        self.pre_trained_embed_weights, self.finetune)
        self.cnn = CNNLayer(self.embedding_dim, self.kernel_size, self.kernel_num)
        self.linear = LinearLayer(self.cnn.output_dim, self.label_num, True, self.linear_activation)

    def forward(self, inputs=None, labels=None, **kwargs):
        x1 = self.embedding(inputs)
        x1 = x1.unsqueeze(1)  # add channel

        x2 = self.cnn(x1)
        x2 = self.dropout(x2)
        x2 = x2.squeeze(-1)  # drop channel

        x3 = self.linear(x2)
        logits = self.softmax(x3)
        outputs = (logits,)

        if labels is not None:
            loss = self.loss(logits, labels)
            outputs = outputs + (loss,)
        else:
            outputs = outputs + (None,)
        return outputs  # logits, loss

import torch
import torch.nn as nn

from ...layers import BertLayer, LinearLayer


class BertClassificationModel(nn.Module):
    def __init__(self,
                 label_num,
                 hidden_dim,
                 finetune=True,
                 pre_trained_model_path=None,
                 linear_activation=None,
                 dropout_rate=0.3
                 ):

        super(BertClassificationModel, self).__init__()
        self.label_num = label_num
        self.hidden_dim = hidden_dim
        self.finetune = finetune
        self.pre_trained_model_path = pre_trained_model_path

        self.dropout_rate = dropout_rate

        self.dropout = nn.Dropout(self.dropout_rate)
        self.softmax = torch.nn.Softmax(dim=1)
        self.loss = torch.nn.MSELoss()

        self.bert = BertLayer(pre_trained_model_path=pre_trained_model_path, finetune=finetune)
        self.linear1 = LinearLayer(input_dim=self.bert.config.hidden_size,
                                   output_dim=hidden_dim, bias=True, activation=linear_activation)
        self.linear2 = LinearLayer(input_dim=hidden_dim, output_dim=label_num,
                                   bias=True, activation=linear_activation)

    def forward(self, inputs=None, labels=None, mask=None, **kwargs):
        if not self.finetune:
            self.bert.eval()
            with torch.no_grad():
                _, x1 = self.bert(input_ids=inputs, attention_mask=mask)
        else:
            _, x1 = self.bert(input_ids=inputs, attention_mask=mask)

        x1 = self.dropout(x1)

        x2 = self.linear1(x1)
        x2 = self.dropout(x2)

        x3 = self.linear2(x2)
        x3 = self.softmax(x3)

        outputs = (x3,)

        if labels is not None:
            loss = self.loss(x3, labels)
            outputs = outputs + (loss,)
        else:
            outputs = outputs + (None, )
        return outputs  # logits, loss
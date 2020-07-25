from torch import nn


class LinearLayer(nn.Module):
    """
    This layer is callable for fully-connected layer with initialized weight and activation function
    """
    def __init__(self, input_dim, output_dim, bias=True, activation=None):
        """
        :param input_dim: input dim (int)
        :param output_dim: output dim (int)
        :param bias: if use bias with linear function, default True (boolean)
        :param activation:  activation function, default ReLU (torch.nn func)
        """
        super(LinearLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.linear = nn.Linear(input_dim, output_dim, bias)
        self.activation = nn.ReLU() if activation is None else activation()

        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, sentence):
        sent_inputs = self.linear(sentence)  # shape: (batch, seq_len, word_vec_size)
        return sent_inputs

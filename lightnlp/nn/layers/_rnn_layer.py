import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RNNLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, bidirectional=True, batch_first=True, cell_type='lstm',
                 batch_size=1):
        super(RNNLayer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.output_dim = hidden_dim * self.num_directions
        self.batch_first = batch_first
        self.cell_type = cell_type
        self.batch_size = batch_size

        if self.cell_type.lower() == 'lstm':
            self.rnn = nn.LSTM(input_size=self.input_dim,
                               hidden_size=self.hidden_dim // self.num_directions,
                               num_layers=self.num_layers,
                               bidirectional=self.bidirectional,
                               batch_first=self.batch_first)
        elif self.cell_type.lower() == 'gru':
            self.rnn = nn.GRU(input_size=self.input_dim,
                              hidden_size=self.hidden_dim // self.num_directions,
                              num_layers=self.num_layers,
                              bidirectional=self.bidirectional,
                              batch_first=self.batch_first)

    def init_hidden(self):
        if self.cell_type.lower() == 'lstm':
            return (
                Variable(torch.zeros(self.num_layers * self.num_directions, self.batch_size,
                                     self.hidden_dim // self.num_directions)),
                Variable(torch.zeros(self.num_layers * self.num_directions, self.batch_size,
                                     self.hidden_dim // self.num_directions))
            )
        elif self.cell_type.lower() == 'gru':
            return Variable(torch.zeros(self.num_layers * self.num_directions, self.batch_size, self.hidden_dim))
        return None

    def forward(self, inputs, mask=None, hidden_state=None, return_type="one"):
        if return_type not in ("one", "all"):
            raise ValueError("return_type must be \'one\' or \'all\'!")
        if hidden_state is None:
            hidden_state = self.init_hidden()
        if mask is None:
            output, hidden = self.rnn(inputs, hidden_state)
            if return_type == "one":
                return output[:, -1, :]
            return output

        else:
            length = [int(sum(i)) for i in mask]
            inputs = pack_padded_sequence(input=inputs, lengths=length, batch_first=self.batch_first)
            output, hidden = self.rnn(inputs, hidden_state)
            output, length = pad_packed_sequence(output, batch_first=self.batch_first)
            if return_type == "one":
                return torch.stack([x[i - 1] for i, x in zip(length, output)], dim=0)
            else:
                return output


if __name__ == "__main__":
    data = torch.tensor([
        [[1., 1., 1., 1.], [1., 1., 1., 1.], [1., 1., 1., 1.]],
        [[1., 1., 1., 1.], [1., 1., 1., 1.], [0., 0., 0., 0.]],
        [[1., 1., 1., 1.], [1., 1., 1., 1.], [0., 0., 0., 0.]],
    ])
    mask = [[1, 1, 1], [1, 1, 0], [1, 1, 0]]
    rnn = RNNLayer(input_dim=4, hidden_dim=2, num_layers=1, batch_first=True, batch_size=3)
    output = rnn(inputs=data, mask=mask, return_type="all")
    print(output)

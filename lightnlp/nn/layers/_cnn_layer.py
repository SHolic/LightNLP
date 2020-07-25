import torch
import torch.nn as nn


class CNNLayer(nn.Module):
    """
        This layer is callable for 1d convolution and pooling functions with fatten result
    """
    def __init__(self,
                 input_dim,
                 kernel_size=(3, 4, 5),
                 kernel_num=200):
        """
        :param input_dim: input dim (type:int)
        :param kernel_size: kernel size of convolution, default is (3,4,5) (type:tuple or list)
        :param kernel_num: channel of each kernel, default is 200 (type:int)
        """
        super(CNNLayer, self).__init__()
        self.output_dim = len(kernel_size) * kernel_num

        self.convolutions = nn.ModuleList(
            [nn.Conv2d(1, kernel_num, (ks, input_dim)) for ks in kernel_size]
        )

    def forward(self, x):
        con_ret = [c(x) for c in self.convolutions]
        pooling_x = [nn.functional.max_pool1d(c.squeeze(-1), c.size()[2]) for c in con_ret]
        flat_pool = torch.cat(pooling_x, 1)
        return flat_pool  # (batch, len(kernel_size)*kernel_num)

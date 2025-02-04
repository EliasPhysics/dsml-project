import torch
import torch.nn as nn
import math
from torch import nn, Tensor


class Encoder(nn.Module):
    """
    This class encodes the position encoded input sequence to a representation in latent space
    """

    def __init__(self):

        """
        Parameters:
            max_seq_len: the maximum length of the input sequences. This is necessary for positional encoding
            d_model: The dimension of the data dimensoin
                     (Vaswani et al, 2017)
        """
        # initialize the torch nn.Module
        super().__init__()




    def forward(self, x: Tensor) -> Tensor:
        """
        Inputshape:


        Returns tensor of shape [batch_size, max_seq_len, d_model]
        """
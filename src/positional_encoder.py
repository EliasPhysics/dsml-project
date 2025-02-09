import torch
import torch.nn as nn
import math
from torch import nn, Tensor


class PositionalEncoder(nn.Module):
    """
    This class encodes the position to the input sequence.
    """

    def __init__(self, max_seq_len: int = 5000, d_model: int = 512, dropout: float = 0.2):

        """
        Parameters:
            max_seq_len: the maximum length of the input sequences. This is necessary for positional encoding
            d_model: The dimension of the input layer output
        """
        # initialize the torch nn.Module
        super().__init__()

        # save parameters to class
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)

        # create array for positional encoding
        position_counter = torch.arange(max_seq_len).unsqueeze(1)
        # taken from the positional encoding torch tutorial
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        # create positional encoding shift to add to the sequential data
        self.pos_encoding = torch.zeros(1, max_seq_len, d_model)
        self.pos_encoding[0, :, 0::2] = torch.sin(position_counter * div_term)
        self.pos_encoding[0, :, 1::2] = torch.cos(position_counter * div_term)

        # this makes torch register the positional encoding as non-trainable parameter
        self.register_buffer('pe', self.pos_encoding)


    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, enc_seq_len, dim_val]

        returns: Tensor, shape [batch_size,enc_seq_len, dim_val]
        """
        print(f"forward tensor shape: {x.shape}")
        x = x + self.pos_encoding[:, :x.size(1)]

        return self.dropout(x)
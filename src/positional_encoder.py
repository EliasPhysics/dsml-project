import torch
import torch.nn as nn
import math
from torch import nn, Tensor


class PositionalEncoder(nn.Module):
    """
    This class encodes the position to the input sequence.
    """

    def __init__(self, max_seq_len: int = 1000, d_model: int = 128, dropout: float = 0.2):

        """
        Parameters:
            max_seq_len: the maximum length of the input sequences. This is necessary for positional encoding
            d_model: The dimension of the input layer output
        """
        # initialize the torch nn.Module
        super().__init__()




    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, enc_seq_len, dim_val]

        returns: Tensor, shape [batch_size,enc_seq_len, dim_val]
        """
        #print(f"forward tensor shape: {x.shape}, pos enc shape: {self.pe[:, :x.size(1)].shape}")
        x = x + self.pe[:,:x.size(1)]

        return self.dropout(x)
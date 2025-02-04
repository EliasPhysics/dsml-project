import torch
import torch.nn as nn
import math
from torch import nn, Tensor


class Decoder(nn.Module):
    """
    This class decodes
    """

    def __init__(self,  dim_feedforward_decoder: int=2048,num_predicted_features: int=1):

        """
        Parameters:
            max_seq_len: the maximum length of the input sequences. This is necessary for positional encoding
            d_model: The dimension of the data dimensoin
                     (Vaswani et al, 2017)
        """
        # initialize the torch nn.Module
        super().__init__()

        self.dec_seq_len = dec_seq_len

        self.decoder_input_layer = nn.Linear(
            in_features=num_predicted_features,
            out_features=dim_val
        )
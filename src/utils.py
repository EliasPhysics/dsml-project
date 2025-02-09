import os
import numpy as np
import pandas as pd
import torch
from torch import Tensor

def generate_square_subsequent_mask(dim1: int, dim2: int):
    """
    Generates an upper-triangular matrix of -inf, with zeros on diag.
    Modified from:
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html

    Args:

        dim1: int, for both src and tgt masking, this must be target sequence
              length

        dim2: int, for src masking this must be encoder sequence length (i.e.
              the length of the input sequence to the model),
              and for tgt masking, this must be target sequence length
    Return:

        A Tensor of shape [dim1, dim2]
    """
    return torch.triu(torch.ones(dim1, dim2) * float('-inf'), diagonal=1)


def get_indices_input_target(num_obs, input_len, step_size, forecast_horizon, target_len):
    """
    Produce all the start and end index positions of all sub-sequences.
    The indices will be used to split the data into sub-sequences on which
    the models will be trained.

    Returns a tuple with four elements:
    1) The index position of the first element to be included in the input sequence
    2) The index position of the last element to be included in the input sequence
    3) The index position of the first element to be included in the target sequence
    4) The index position of the last element to be included in the target sequence
    """
    stop_position = num_obs - 1

    subseq_start_index =0
    subseq_stop_index = input_len

    tgt_start_index = subseq_stop_index + forecast_horizon
    tgt_stop_index = tgt_start_index + target_len

    print("target_last_idx is {}".format(tgt_start_index))
    print("stop_position is {}".format(stop_position))
    indices = []
    while tgt_stop_index <= stop_position:
        indices.append((subseq_start_index, subseq_stop_index, tgt_start_index,  tgt_stop_index))
        subseq_start_index += step_size
        subseq_stop_index += step_size
        target_first_idx = (subseq_stop_index+ forecast_horizon)
        tgt_stop_index = target_first_idx + target_len

    return indices


def get_indices_entire_sequence(data: pd.DataFrame, window_size: int, step_size: int) -> list:

    start_index = 0
    stop_index = window_size

    list_indices = []

    # go through all available timeframes
    while stop_index <=  len(data) - 1:
        list_indices.append((start_index, stop_index))

        start_index += step_size
        stop_index += step_size

    return list_indices


def read_data(data_dir:str) -> Tensor:

    array = np.load(data_dir)
    print(f"Training data shape: {array.shape}")

    return torch.from_numpy(array)



if __name__ == '__main__':
    os.chdir("..")
    read_data("data/lorenz63_on0.05_train.npy")
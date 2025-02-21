import os
import numpy as np
import torch
from torch import Tensor

def generate_square_subsequent_mask(dim1: int, dim2: int):
    return torch.triu(torch.ones(dim1, dim2) * float('-inf'), diagonal=1)


def get_indices_entire_sequence(data: Tensor, window_size: int, step_size: int) -> list:

    start_index = 0
    stop_index = window_size

    list_indices = []

    # go through all available timeframes
    while stop_index < len(data):
        list_indices.append((start_index, stop_index))

        start_index += step_size
        stop_index += step_size

    return list_indices


def read_data(data_dir:str) -> Tensor:

    array = np.load(data_dir)
    print(f"Loaded data shape: {array.shape}")

    return torch.from_numpy(array)



if __name__ == '__main__':
    os.chdir("..")
    read_data("data/lorenz63_on0.05_train.npy")
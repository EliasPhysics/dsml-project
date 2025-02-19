import numpy
import torch
import training
from tqdm import tqdm
from dataset import TransformerDataset
import utils
from model import TimeSeriesTransformer
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import random
import numpy as np

# Set device
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')


os.chdir("..")
data_validation = utils.read_data("data/lorenz63_test.npy")
input_size = data_validation.shape[1]

warmup_steps = 25

## Params
dim_val = 512
n_heads = 8
n_decoder_layers = 4
n_encoder_layers = 4
dec_seq_len = 92 # length of input given to decoder
enc_seq_len = 153 # length of input given to encoder
output_seq_len = 64 # target sequence length. If hourly data and length = 48, you predict 2 days ahead
window_size = enc_seq_len + output_seq_len # used to slice data into sub-sequences
step_size = 10 # Step size, i.e. how many time steps does the moving window move at each step
in_features_encoder_linear_layer = 2048
in_features_decoder_linear_layer = 2048
max_seq_len = enc_seq_len


# Load the trained model
model_path = "models/test4.pth"  # Change path if needed
model = TimeSeriesTransformer(input_size=input_size,
                              dec_seq_len=enc_seq_len
                              ).to(device)

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set to evaluation mode
    print("Model loaded successfully!")
else:
    raise FileNotFoundError(f"Model file not found at {model_path}")



data_train = utils.read_data("data/lorenz63_on0.05_train.npy")

start = random.randrange(len(data_train)-enc_seq_len-output_seq_len)
initial_condition = data_train[start:start+enc_seq_len + output_seq_len]
initial_condition = initial_condition.unsqueeze(0)
print(initial_condition.shape)

src = initial_condition[:,:enc_seq_len]
tgt = initial_condition[:,enc_seq_len - 1:len(initial_condition) - 1]


# Run inference on the test set
print("Creating data to compare")

warmup_time_series = initial_condition

i = 0

while i < warmup_steps:
    with torch.no_grad():
        output = model(src, tgt)
        warmup_time_series = torch.cat((warmup_time_series, output), dim=1)

        src = warmup_time_series[:,output.shape[1]:enc_seq_len]
        tgt = warmup_time_series[:,output.shape[1] + enc_seq_len - 1:len(warmup_time_series) - 1]
    i += 1





generated_time_series = torch.zeros_(1,T,3)


with torch.no_grad():
    while len(generated_time_series) < len(data_validation):
        with torch.no_grad():
            output = model(src, tgt)
            warmup_time_series = torch.cat((warmup_time_series, output), dim=1)

            src = warmup_time_series[:, output.shape[1]:enc_seq_len]
            tgt = warmup_time_series[:, output.shape[1] + enc_seq_len - 1:len(warmup_time_series) - 1]



# Plot the loss curve
print("Testing complete!")

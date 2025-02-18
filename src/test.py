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

# Set device
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

os.chdir("..")
data_validation = utils.read_data("data/lorenz63_on0.05_test.npy")
input_size = data_validation.shape[1]
batch_size = 32



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


data_train = utils.read_data("data/lorenz63_on0.05_train.npy")

start = random.randrange(len(data_train)-enc_seq_len-dec_seq_len)
initial_condition = data_train[start:start+enc_seq_len + dec_seq_len]

src = initial_condition[:enc_seq_len]
tgt = initial_condition[enc_seq_len:]



# Load the trained model
model_path = "models/test1.pth"  # Change path if needed
model = TimeSeriesTransformer(input_size=input_size,
                              dec_seq_len=enc_seq_len
                              ).to(device)

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set to evaluation mode
    print("Model loaded successfully!")
else:
    raise FileNotFoundError(f"Model file not found at {model_path}")

# Load test dataset
training_indices = utils.get_indices_entire_sequence(
    data=data_validation,
    window_size=window_size,
    step_size=step_size)


val_data_loader = TransformerDataset(data=data_validation,
                                 indices=training_indices,
                                 enc_seq_len=enc_seq_len,
                                 dec_seq_len=dec_seq_len,
                                 target_seq_len=output_seq_len)
val_data_loader = DataLoader(val_data_loader, batch_size)


# Loss function
criterion = torch.nn.MSELoss()

# Initialize lists to store loss values
all_losses = []
true_values = []
predictions = []
generated_time_series = numpy.zeros_like(data_validation) # val set size

# Run inference on the test set
print("Creating data to compare")
with torch.no_grad():
    while len(generated_time_series) < len(data_validation):

        src, tgt = src.to(device), tgt.to(device)
        src_mask, tgt_mask = src_mask.to(device), tgt_mask.to(device)

        # Forward pass
        src = initial_condition[:enc_seq_len]
        tgt= initial_condition[enc_seq_len:enc_seq_len+dec_seq_len]
        output = model(src=src, tgt=tgt)

        # Compute loss
        predictions.append(output.cpu().numpy())

# Plot the loss curve
print("Testing complete!")

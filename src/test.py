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

warmup_steps = 100

## Params
dim_val = 512
n_heads = 8
n_decoder_layers = 4
n_encoder_layers = 4
dec_seq_len = 16 # length of input given to decoder
enc_seq_len = 32 # length of input given to encoder
output_seq_len = 8 # target sequence length. If hourly data and length = 48, you predict 2 days ahead
window_size = enc_seq_len + output_seq_len # used to slice data into sub-sequences
step_size = 10 # Step size, i.e. how many time steps does the moving window move at each step
in_features_encoder_linear_layer = 2048
in_features_decoder_linear_layer = 2048
max_seq_len = enc_seq_len


# Load the trained model
model_name = "test5"
model_path = f"models/{model_name}.pth"  # Change path if needed
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
        warmup_time_series = torch.cat((warmup_time_series, output[:,-1].unsqueeze(0)), dim=1)
        #warmup_time_series = warmup_time_series[:,output.shape[1]:]
        warmup_time_series = warmup_time_series[:, 1:]
        #print(output,tgt)
        src = warmup_time_series[:,:enc_seq_len]
        tgt = warmup_time_series[:,enc_seq_len - 1:warmup_time_series.shape[1] - 1]
    i += 1





generated_time_series = warmup_time_series
current_generated_time_series = warmup_time_series
src = warmup_time_series[:,:enc_seq_len]
tgt = warmup_time_series[:,enc_seq_len - 1:len(warmup_time_series) - 1]

with torch.no_grad():
    while generated_time_series.shape[1] < data_validation.shape[0]:
        output = model(src, tgt)
        generated_time_series = torch.cat((generated_time_series, output[:,-1].unsqueeze(0)), dim=1)
        current_generated_time_series = torch.cat((current_generated_time_series, output[:,-1].unsqueeze(0)), dim=1)
        #current_generated_time_series = current_generated_time_series[:,output.shape[1]:]
        current_generated_time_series = current_generated_time_series[:, 1:]

        src = current_generated_time_series[:, :enc_seq_len]
        tgt = current_generated_time_series[:, enc_seq_len - 1:current_generated_time_series.shape[1] - 1]




generated_time_series = generated_time_series[:,:data_validation.shape[0]]


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create a 3D figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x,y,z = torch.transpose(generated_time_series.squeeze(0),dim0=0,dim1=1)
x_or,y_or,z_or = torch.transpose(data_validation,dim0=0,dim1=1)

# Plot the trajectory
ax.plot(x, y, z, label='3D Trajectory', color='b',alpha=0.7)
ax.plot(x_or, y_or, z_or, label='Original Trajectory', color='r',alpha=0.2)

# Labels and title
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("3D Trajectory Plot")

# Show the legend
ax.legend()

# Display the plot
plt.savefig(f"plots/3D_Trajectory_model{model_name}.png")
plt.show()


print("Data generation complete!")

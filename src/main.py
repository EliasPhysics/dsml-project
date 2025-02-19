import torch
import training
from tqdm import tqdm
from dataset import TransformerDataset
import utils
from model import TimeSeriesTransformer
import os
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')


# Training parameters
epochs = 5
batch_size = 32


# initialize data
os.chdir("..")
data = utils.read_data("data/lorenz63_on0.05_train.npy")
input_size = data.shape[1]




## Params
dim_val = 256
n_heads = 8
n_decoder_layers = 2
n_encoder_layers = 2
dec_seq_len = 92 # length of input given to decoder
enc_seq_len = 153 # length of input given to encoder
output_seq_len = 64 # target sequence length. If hourly data and length = 48, you predict 2 days ahead
window_size = enc_seq_len + output_seq_len # used to slice data into sub-sequences

step_size = 100 # Step size, i.e. how many time steps does the moving window move at each step

in_features_encoder_linear_layer = 2048
in_features_decoder_linear_layer = 2048
max_seq_len = enc_seq_len




training_indices = utils.get_indices_entire_sequence(
    data=data,
    window_size=window_size,
    step_size=step_size)

training_data = TransformerDataset(data=data,
                                 indices=training_indices,
                                 enc_seq_len=enc_seq_len,
                                 dec_seq_len=dec_seq_len,
                                 target_seq_len=output_seq_len)

training_data = DataLoader(training_data, batch_size)

model = TimeSeriesTransformer(input_size=input_size,
                              dec_seq_len=enc_seq_len
                              )

optimizer = torch.optim.Adam(params=model.parameters())
criterion = torch.nn.HuberLoss()

# Make src mask for decoder with size:
# [batch_size*n_heads, output_sequence_length, enc_seq_len]
src_mask = utils.generate_square_subsequent_mask(
    dim1=output_seq_len,
    dim2=enc_seq_len
    )

# Make tgt mask for decoder with size:
# [batch_size*n_heads, output_sequence_length, output_sequence_length]
tgt_mask = utils.generate_square_subsequent_mask(
    dim1=output_seq_len,
    dim2=output_seq_len
    )

losses = []

# Iterate over all epochs
for epoch in tqdm(range(epochs)):

    # Iterate over all (x,y) pairs in training dataloader
    for i, (src, tgt, tgt_y) in enumerate(training_data):
        # zero the parameter gradients
        optimizer.zero_grad()
        #print(src.shape, tgt.shape)

        # Generate masks
        src_mask = utils.generate_square_subsequent_mask(
            dim1=output_seq_len,
            dim2=enc_seq_len
        )

        tgt_mask = utils.generate_square_subsequent_mask(
            dim1=output_seq_len,
            dim2=output_seq_len
        )

        # Make forecasts
        #print(f"src: {src.shape}, tgt: {tgt.shape}")
        prediction = model(src=src, tgt=tgt, src_mask=src_mask, tgt_mask=tgt_mask)

        # Compute and backprop loss
        loss = criterion(tgt_y, prediction)
        losses.append(loss.detach())

        loss.backward()
        #print(loss.detach())

        # Take optimizer step
        optimizer.step()

    # Iterate over all (x,y) pairs in validation dataloader
    model.eval()

model_name = "test4"
torch.save(model.state_dict(), f"models/{model_name}.pth")

plt.plot(range(len(losses)),losses)
plt.ylabel("loss")
plt.xlabel("epochs")
plt.savefig(f"plots/training_{model_name}.png")


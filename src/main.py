import torch
import training
from dataset import TransformerDataset
import utils
from model import TimeSeriesTransformer
import os
from torch.utils.data import DataLoader

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')


# Training parameters
epochs = 10
batch_size = 32
forecast_window = 10


# initialize data
os.chdir("..")
data = utils.read_data("data/lorenz63_on0.05_train.npy")
input_size = data.shape[1]




## Params
dim_val = 512
n_heads = 8
n_decoder_layers = 4
n_encoder_layers = 4
dec_seq_len = 92 # length of input given to decoder
enc_seq_len = 153 # length of input given to encoder
output_seq_len = 48 # target sequence length. If hourly data and length = 48, you predict 2 days ahead
window_size = enc_seq_len + output_seq_len # used to slice data into sub-sequences
step_size = 10 # Step size, i.e. how many time steps does the moving window move at each step
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

i, batch = next(enumerate(training_data))

src, trg, trg_y = batch

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

output = model(
    src=src,
    tgt=trg,
    src_mask=src_mask,
    tgt_mask=tgt_mask
    )


# Iterate over all epochs
for epoch in range(epochs):

    # Iterate over all (x,y) pairs in training dataloader
    for i, (src, tgt, tgt_y) in enumerate(training_data):
        # zero the parameter gradients
        optimizer.zero_grad()

        # Generate masks
        src_mask = utils.generate_square_subsequent_mask(
            dim1=enc_seq_len,
            dim2=enc_seq_len
        )

        tgt_mask = utils.generate_square_subsequent_mask(
            dim1=enc_seq_len,
            dim2=output_seq_len
        )

        # Make forecasts
        #print(f"src: {src.shape}, tgt: {tgt.shape}")
        prediction = model(src, tgt, src_mask, tgt_mask)

        # Compute and backprop loss
        loss = criterion(tgt_y, prediction)

        loss.backward()
        #print(loss)

        # Take optimizer step
        optimizer.step()

    # Iterate over all (x,y) pairs in validation dataloader
    model.eval()

    #data_validation = utils.read_data("data/lorenz63_on0.05_train.npy")

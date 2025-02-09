import torch
import training
from dataset import TransformerDataset
import utils
from model import TimeSeriesTransformer
import os


# Training parameters
epochs = 10
forecast_window = 10
enc_seq_len = 20
dec_seq_len = 20
input_len = 40
d_model = 512
step_size = 5
forecast_horizon = 5
target_len = 20
n_enc_layers = 2
n_dec_layers = 2


# initialize data
os.chdir("..")
data = utils.read_data("data/lorenz63_on0.05_train.npy")
input_size = data.shape[1]


# create training batches
indices = utils.get_indices_input_target(num_obs=data.shape[0],
                                         input_len=input_len,
                                         step_size=step_size,
                                         forecast_horizon=forecast_horizon,
                                         target_len=target_len)

#masks = utils.generate_square_subsequent_mask()

datamanager = TransformerDataset(data=data,
                                 indices=indices,
                                 enc_seq_len=enc_seq_len,
                                 dec_seq_len=dec_seq_len,
                                 target_seq_len=target_len)


model = TimeSeriesTransformer(input_size=input_size,
                              d_model=d_model,
                              n_encoder_layers=n_enc_layers,
                              n_decoder_layers=n_dec_layers,
                              dec_seq_len=dec_seq_len
                              )

optimizer = torch.optim.Adam(params=model.parameters())
criterion = torch.nn.HuberLoss()


# Iterate over all epochs
for epoch in range(epochs):

    # Iterate over all (x,y) pairs in training dataloader
    for i, (src, tgt, tgt_y) in enumerate(datamanager):
        # zero the parameter gradients
        optimizer.zero_grad()

        # Generate masks
        tgt_mask = utils.generate_square_subsequent_mask(
            dim1=forecast_window,
            dim2=forecast_window
        )

        src_mask = utils.generate_square_subsequent_mask(
            dim1=forecast_window,
            dim2=enc_seq_len
        )

        # Make forecasts
        prediction = model(src, tgt, src_mask, tgt_mask)

        # Compute and backprop loss
        loss = criterion(tgt_y, prediction)

        loss.backward()

        # Take optimizer step
        optimizer.step()

    # Iterate over all (x,y) pairs in validation dataloader
    model.eval()

    with torch.no_grad():

        for i, (src, _, tgt_y) in enumerate(validation_dataloader):
            prediction = inference.run_encoder_decoder_inference(
                model=model,
                src=src,
                forecast_window=forecast_window,
                batch_size=src.shape[1]
            )

            loss = criterion(tgt_y, prediction)
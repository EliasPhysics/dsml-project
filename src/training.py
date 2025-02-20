import torch
from tqdm import tqdm
from dataset import TransformerDataset
import utils
from model import TimeSeriesTransformer
import os
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import hyperparameters
import csv

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')


def train_TimeSeriesTransformer(data_path, args):
    # Training parameters
    epochs = args["epochs"]
    batch_size = args["batch_size"]

    # Initialize data
    data = utils.read_data(data_path)

    ## Params from args
    dec_seq_len = args["dec_seq_len"]
    enc_seq_len = args["enc_seq_len"]
    output_seq_len = args["output_seq_len"]
    window_size = args["window_size"]
    step_size = args["step_size"]


    training_indices = utils.get_indices_entire_sequence(
        data=data,
        window_size=window_size,
        step_size=step_size)

    training_data = TransformerDataset(data=data,
                                     indices=training_indices,
                                     enc_seq_len=enc_seq_len,
                                     dec_seq_len=dec_seq_len,
                                     target_seq_len=output_seq_len)

    training_data = DataLoader(training_data, batch_size,shuffle=True)

    model = TimeSeriesTransformer(
        input_size=data.shape[1],  # Assuming 'data' is already loaded
        dec_seq_len=args["dec_seq_len"],
        d_model=args["dim_val"],
        n_encoder_layers=args["n_encoder_layers"],
        n_decoder_layers=args["n_decoder_layers"],
        dropout=0.2,
        max_seq_len=args["max_seq_len"],
        dim_feedforward_encoder=args["in_features_encoder_linear_layer"],
        n_heads=args["n_heads"],
        dim_feedforward_decoder=args["in_features_decoder_linear_layer"],
        num_predicted_features= 3 # Assuming prediction targets match input features
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



    model_name = args["model_name"]
    torch.save(model.state_dict(), f"models/{model_name}.pth")

    plt.plot(range(len(losses)),losses)
    plt.ylabel("loss")
    plt.xlabel("epochs")
    plt.savefig(f"plots/training_{model_name}.png")

    # Save hyperparameters as CSV
    with open(f"models/{model_name}.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "Age"])  # Column headers
        for key, value in args.items():
            writer.writerow([key, value])


if __name__=="__main__":
    os.chdir("..")
    data_path = "data/lorenz63_on0.05_train.npy"
    train_TimeSeriesTransformer(data_path=data_path,args=hyperparameters.args)
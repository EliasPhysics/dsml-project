import torch
import training
from tqdm import tqdm
from dataset import TransformerDataset
import utils
from model import TimeSeriesTransformer
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Set device
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

os.chdir("..")
data_validation = utils.read_data("data/lorenz63_on0.05_train.npy")
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

# Run inference on the test set
print("Running inference on test data...")
with torch.no_grad():
    for batch in tqdm(data_validation):
        src, tgt, tgt_y = batch

        src, tgt = src.to(device), tgt.to(device)
        src_mask, tgt_mask = src_mask.to(device), tgt_mask.to(device)

        # Forward pass
        output = model(src, tgt, src_mask, tgt_mask)

        # Compute loss
        loss = criterion(output, tgt)
        all_losses.append(loss.item())

        # Store true values and predictions for plotting
        true_values.append(tgt.cpu().numpy())
        predictions.append(output.cpu().numpy())

# Plot the loss curve
plt.figure(figsize=(10, 5))
plt.plot(all_losses, label="Test Loss", marker='o', linestyle='-')
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.title("Test Loss Curve")
plt.legend()
plt.show()

print("Testing complete!")

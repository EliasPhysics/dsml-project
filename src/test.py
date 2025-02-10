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

# Define model parameters (ensure these match the training phase)
d_model = 512
n_heads = 8
num_encoder_layers = 6
num_decoder_layers = 6
dim_feedforward = 2048
dropout = 0.1
num_predicted_features = 3  # Adjust based on your dataset

# Load the trained model
model_path = "saved_model.pth"  # Change path if needed
model = TimeSeriesTransformer(
    d_model=d_model,
    n_heads=n_heads,
    num_encoder_layers=num_encoder_layers,
    num_decoder_layers=num_decoder_layers,
    dim_feedforward=dim_feedforward,
    dropout=dropout,
    num_predicted_features=num_predicted_features
).to(device)

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set to evaluation mode
    print("Model loaded successfully!")
else:
    raise FileNotFoundError(f"Model file not found at {model_path}")

# Load test dataset
test_dataset = TransformerDataset(split="test")  # Ensure your dataset class supports test split
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Loss function
criterion = torch.nn.MSELoss()

# Initialize lists to store loss values
all_losses = []
true_values = []
predictions = []

# Run inference on the test set
print("Running inference on test data...")
with torch.no_grad():
    for batch in tqdm(test_loader):
        src, tgt, src_mask, tgt_mask = batch

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

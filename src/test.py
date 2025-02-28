import torch
from tqdm import tqdm
import utils
from model import TimeSeriesTransformer
import os
import matplotlib.pyplot as plt
import numpy as np
import random
import hyperparameters
from psd import power_spectrum_error, get_average_spectrum

# Set device
def generate_TimeSeriesTransformer(data_path_train,data_path,args):

    # data for comparison/ validation
    data_validation = utils.read_data(data_path)

    # numver of warmup steps
    warmup_steps = 100

    model_name = args["model_name"]
    enc_seq_len = args["enc_seq_len"]
    output_seq_len = args["output_seq_len"]

    data_train = utils.read_data(data_path_train)

    # Load the trained model
    model_path = f"models/{model_name}.pth"  # Change path if needed
    model = TimeSeriesTransformer(
        input_size=data_validation.shape[1],  # Assuming 'data' is already loaded
        dec_seq_len=args["dec_seq_len"],
        d_model=args["dim_val"],
        n_encoder_layers=args["n_encoder_layers"],
        n_decoder_layers=args["n_decoder_layers"],
        dropout=0.2,  # Default value
        max_seq_len=args["max_seq_len"],
        dim_feedforward_encoder=args["in_features_encoder_linear_layer"],
        n_heads=args["n_heads"],
        dim_feedforward_decoder=args["in_features_decoder_linear_layer"],
        num_predicted_features= data_validation.shape[1]  # Assuming prediction targets match input features
    )

    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

    # load trained model
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()  # Set to evaluation mode
        print("Model loaded successfully!")
    else:
        raise FileNotFoundError(f"Model file not found at {model_path}")



    # take some random initial condition from the training set
    start = random.randrange(len(data_train)-enc_seq_len-output_seq_len-1)
    initial_condition = data_train[start:start+enc_seq_len + output_seq_len]
    initial_condition = initial_condition.unsqueeze(0)

    print("Creating new time series")


    # first warm up the model to get away from initial condition
    warmup_time_series = initial_condition

    with torch.no_grad():
        for _ in range(warmup_steps):
            # devide data into src and tgt (encoder and decoder input)
            src = warmup_time_series[:, :enc_seq_len]
            tgt = warmup_time_series[:, enc_seq_len - 1:warmup_time_series.shape[1] - 1]

            # calculate output
            output = model(src=src, tgt=tgt)


            warmup_time_series = torch.cat((warmup_time_series, output[:,-1].unsqueeze(0)), dim=1)
            warmup_time_series = warmup_time_series[:, 1:]


    generated_time_series = warmup_time_series
    current_generated_time_series = warmup_time_series

    with torch.no_grad():
        for _ in tqdm(range(int(data_validation.shape[0]))):
            src = current_generated_time_series[:, :enc_seq_len]
            tgt = current_generated_time_series[:, enc_seq_len - 1:current_generated_time_series.shape[1] - 1]

            output = model(src=src, tgt=tgt)
            generated_time_series = torch.cat((generated_time_series, output[:,-1].unsqueeze(0)), dim=1)
            current_generated_time_series = torch.cat((current_generated_time_series, output[:,-1].unsqueeze(0)), dim=1)
            current_generated_time_series = current_generated_time_series[:, 1:]


    # cute time series to be the right length
    generated_time_series = generated_time_series[:,:data_validation.shape[0]]



    # Create a 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    generated_trajectory = torch.transpose(generated_time_series.squeeze(0),dim0=0,dim1=1)
    np.save(f"data/generated_trajectory_{model_name}.npy", generated_time_series)

    x,y,z = generated_trajectory[:3]
    x_or,y_or,z_or = torch.transpose(data_validation[:,:3],dim0=0,dim1=1)

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
    #plt.show()
    plt.close()




def analyze_generated_trajectories(model_name,data_validation_path):

    generated_trajectory = utils.read_data(f"data/generated_trajectory_{model_name}.npy")
    test_trajectory = utils.read_data(data_validation_path)

    freqs = np.fft.rfftfreq(generated_trajectory.shape[1])
    idx = np.argsort(freqs)
    ps = get_average_spectrum(generated_trajectory)
    ps_or = get_average_spectrum(test_trajectory.unsqueeze(0))

    num_dims = generated_trajectory.shape[2]  # Number of dimensions

    # Create subplots, one per dimension
    fig, axes = plt.subplots(num_dims, 1, figsize=(8, 4 * num_dims), sharex=True)

    for dim in range(min(num_dims,3)):  # Loop over dimensions , max 3
        ax = axes[dim]
        ax.plot(freqs[idx], ps_or[0,:,dim], label=f'Original (Dim {dim +1})', linestyle="-", color="blue")
        ax.plot(freqs[idx], ps[0,:,dim], label=f'Generated (Dim {dim + 1})', linestyle="--", color="red")
        ax.set_ylabel("Power")
        ax.set_title(f"Power Spectrum (Dimension {dim + 1})")
        ax.legend()
        ax.grid(True)

    # Common labels
    axes[-1].set_xlabel("Frequency")
    plt.suptitle("Power Spectrum Comparison per Dimension")
    plt.tight_layout()

    plt.title(f"Power Spectrum")
    plt.xlabel("Frequency")
    plt.savefig(f"plots/{model_name}_power_spectrum.png")
    plt.close()

    ps_error = power_spectrum_error(generated_trajectory, test_trajectory.unsqueeze(0))
    print(f"The power spectrum error amoounta to; {ps_error}")




if __name__=="__main__":
    os.chdir("..")
    data_path_train = "data/lorenz63_on0.05_train.npy"
    data_path = "data/lorenz63_test.npy"
    generate_TimeSeriesTransformer(data_path=data_path,data_path_train=data_path_train,args=hyperparameters.args63)
    analyze_generated_trajectories(model_name=hyperparameters.args63["model_name"], data_validation_path=data_path)

    data_path_train = "data/lorenz96_on0.05_train.npy"
    data_path = "data/lorenz96_test.npy"
    generate_TimeSeriesTransformer(data_path=data_path, data_path_train=data_path_train, args=hyperparameters.args96)
    analyze_generated_trajectories(model_name=hyperparameters.args96["model_name"], data_validation_path=data_path)
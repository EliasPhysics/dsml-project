
# Define the args dictionary with values
args63 = {
    "model_name": "Lorenz63",
    "epochs": 5,
    "batch_size": 32,
    "dim_val": 256,
    "n_heads": 8,
    "n_decoder_layers": 2,
    "n_encoder_layers": 2,
    "dec_seq_len": 64,  # length of input given to decoder
    "enc_seq_len": 128,  # length of input given to encoder
    "output_seq_len": 32,  # target sequence length
    "window_size": 128 + 32,  # enc_seq_len + output_seq_len
    "step_size": 10,  # Step size for moving window
    "in_features_encoder_linear_layer": 512,
    "in_features_decoder_linear_layer": 512,
    "max_seq_len": 128,  # Same as enc_seq_len
}


args96 = {
    "model_name": "Lorenz96",
    "epochs": 5,
    "batch_size": 32,
    "dim_val": 256,
    "n_heads": 8,
    "n_decoder_layers": 2,
    "n_encoder_layers": 2,
    "dec_seq_len": 64,  # length of input given to decoder
    "enc_seq_len": 128,  # length of input given to encoder
    "output_seq_len": 32,  # target sequence length
    "window_size": 128 + 32,  # enc_seq_len + output_seq_len
    "step_size": 10,  # Step size for moving window
    "in_features_encoder_linear_layer": 512,
    "in_features_decoder_linear_layer": 512,
    "max_seq_len": 128,  # Same as enc_seq_len
}
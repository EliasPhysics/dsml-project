
# Define the args dictionary with values
args = {
    "model_name": "test8",
    "epochs": 6,
    "batch_size": 32,
    "dim_val": 32,
    "n_heads": 4,
    "n_decoder_layers": 2,
    "n_encoder_layers": 2,
    "dec_seq_len": 16,  # length of input given to decoder
    "enc_seq_len": 32,  # length of input given to encoder
    "output_seq_len": 8,  # target sequence length
    "window_size": 32 + 8,  # enc_seq_len + output_seq_len
    "step_size": 10,  # Step size for moving window
    "in_features_encoder_linear_layer": 128,
    "in_features_decoder_linear_layer": 128,
    "max_seq_len": 32,  # Same as enc_seq_len
}

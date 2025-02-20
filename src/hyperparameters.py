
# Define the args dictionary with values
args = {
    "model_name": "test7",
    "epochs": 5,
    "batch_size": 32,
    "dim_val": 64,
    "n_heads": 8,
    "n_decoder_layers": 4,
    "n_encoder_layers": 4,
    "dec_seq_len": 16,  # length of input given to decoder
    "enc_seq_len": 32,  # length of input given to encoder
    "output_seq_len": 8,  # target sequence length
    "window_size": 32 + 8,  # enc_seq_len + output_seq_len
    "step_size": 25,  # Step size for moving window
    "in_features_encoder_linear_layer": 256,
    "in_features_decoder_linear_layer": 256,
    "max_seq_len": 32,  # Same as enc_seq_len
}

from torch import nn, Tensor
import positional_encoder


class Encoder(nn.Module):
    """
    This class encodes the position encoded input sequence to a representation in latent space
    """

    def __init__(self,
                 input_size:int,
                 d_model:int,
                 n_encoder_layers:int,
                 dropout_encoder:float=0.2,
                 dropout_pos_enc:float=0.2,
                 max_seq_len: int = 5000,
                 dim_feedforward_encoder:int=2048):

        """
        Parameters:
            max_seq_len: the maximum length of the input sequences. This is necessary for positional encoding
            d_model: The dimension of the input layer output
        """
        # initialize the torch nn.Module
        super().__init__()


        self.encoder_input_layer = nn.Linear(
            in_features=input_size,
            out_features=d_model
        )

        # Create positional encoder from other module
        self.positional_encoding_layer = positional_encoder.PositionalEncoder(
            max_seq_len=max_seq_len,
            d_model=d_model,
            dropout=dropout_pos_enc
        )

        # now build encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            dim_feedforward=dim_feedforward_encoder,
            dropout=dropout_encoder,
        )

        # stack encoder layers to obtain Encoder
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_encoder_layers,
            norm=None
        )




    def forward(self, src: Tensor) -> Tensor:
        """
        Returns a tensor of shape:

        [ batch_size, target_sequence_length, ?]
        """
        # Pass through the input layer right before the encoder
        src = self.encoder_input_layer(src)

        # Pass through the positional encoding layer
        pos_encoded_src = self.positional_encoding_layer(src)

        encoder_output = self.encoder(
            src=pos_encoded_src
            )

        return encoder_output
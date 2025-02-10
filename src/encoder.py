from torch import nn, Tensor
import positional_encoder


class Encoder(nn.Module):
    """
    This class encodes the position encoded input sequence to a representation in latent space
    """

    def __init__(self,
                 input_size:int,
                 d_model:int,
                 nhead:int,
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





    def forward(self, src: Tensor) -> Tensor:
        """
        Returns a tensor of shape:

        [ batch_size, target_sequence_length, ?]
        """
        # Pass through the input layer right before the encoder
        src = self.encoder_input_layer(src)

        # Pass through the positional encoding layer
        #src = src.unsqueeze(0)
        pos_encoded_src = self.positional_encoding_layer(src)

        encoder_output = self.encoder(
            src=pos_encoded_src
            )

        return encoder_output
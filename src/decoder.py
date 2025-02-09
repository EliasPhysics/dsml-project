from torch import nn, Tensor


class Decoder(nn.Module):
    """
    This class decodes the latent to give an prediction of the time series
    """

    def __init__(self,
                 d_model:int,
                 n_decoder_layers:int,
                 dec_seq_len:int,
                 dropout_decoder:float=0.2,
                 n_heads:int=8,
                 dim_feedforward_decoder:int=2048,
                 num_predicted_features: int=3):

        """
        Parameters:

        """
        # initialize the torch nn.Module
        super().__init__()

        self.dec_seq_len = dec_seq_len

        self.decoder_input_layer = nn.Linear(
            in_features=num_predicted_features,
            out_features=d_model,
        )

        # create one decoder layer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward_decoder,
            dropout=dropout_decoder,
            batch_first = True
        )

       # stack the decoder layers
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=n_decoder_layers,
            norm=None
        )

        self.linear_mapping = nn.Linear(
            in_features=d_model,
            out_features=num_predicted_features
        )

    def forward(self, tgt: Tensor, src: Tensor, tgt_mask: Tensor=None, src_mask: Tensor=None) -> Tensor:

        tgt = self.decoder_input_layer(tgt)

        decoder_output = self.decoder(
            tgt=tgt,
            memory=src,
            tgt_mask=tgt_mask,
            memory_mask=src_mask
        )

        decoder_output = self.linear_mapping(decoder_output)


        return decoder_output
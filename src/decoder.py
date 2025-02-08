from torch import nn, Tensor


class Decoder(nn.Module):
    """
    This class decodes
    """

    def __init__(self,
                 input_size:int,
                 d_model:int,
                 n_decoder_layers:int,
                 dropout_decoder:float=0.2,
                 dropout_pos_enc:float=0.2,
                 num_predicted_features,
                 n_heads:int=8,
                 dim_feedforward_decoder:int=2048):

        """
        Parameters:
            max_seq_len: the maximum length of the input sequences. This is necessary for positional encoding
            d_model: The dimension of the data dimensoin
        """
        # initialize the torch nn.Module
        super().__init__()

        self.dec_seq_len = dec_seq_len

        self.decoder_input_layer = nn.Linear(
            in_features=num_predicted_features,
            out_features=d_model
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


    def forward(self,tgt: Tensor,src:Tensor, tgt_mask: Tensor,src_mask: Tensor) -> Tensor:

        tgt = self.decoder_input_layer(tgt)

        decoder_output = self.decoder(
            tgt=tgt,
            memory=src,
            tgt_mask=tgt_mask,
            memory_mask=src_mask
        )

        # print("From model.forward(): decoder_output shape after decoder: {}".format(decoder_output.shape))

        # Pass through linear mapping
        decoder_output = self.linear_mapping(decoder_output)  # shape [batch_size, target seq len]
        # print("From model.forward(): decoder_output size after linear_mapping = {}".format(decoder_output.size()))

        return decoder_output
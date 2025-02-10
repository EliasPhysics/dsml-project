from torch import nn, Tensor
import math
import torch


class TimeSeriesTransformer(nn.Module):
    def __init__(self,
                 input_size: int,
                 dec_seq_len: int,
                 d_model: int = 512,
                 n_encoder_layers: int = 4,
                 n_decoder_layers: int = 4,
                 dropout: float = 0.2,
                 max_seq_len: int = 512,
                 dim_feedforward_encoder: int = 2048,
                 n_heads: int = 8,
                 dim_feedforward_decoder: int = 2048,
                 num_predicted_features: int = 3
                 ):

        super().__init__()
        # positional encoder

        self.dropout = nn.Dropout(p=dropout)

        # create array for positional encoding
        position_counter = torch.arange(max_seq_len).unsqueeze(1)
        # taken from the positional encoding torch tutorial
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        # create positional encoding shift to add to the sequential data
        pos_encoding = torch.zeros(1, max_seq_len, d_model)
        pos_encoding[0, :, 0::2] = torch.sin(position_counter * div_term)
        pos_encoding[0, :, 1::2] = torch.cos(position_counter * div_term)

        # this makes torch register the positional encoding as non-trainable parameter
        self.register_buffer('pe', pos_encoding)

        # encoder


        self.encoder_input_layer = nn.Linear(
            in_features=input_size,
            out_features=d_model
        )

        # Create positional encoder from other module

        # now build encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead = n_heads,
            dim_feedforward=dim_feedforward_encoder,
            dropout=dropout,
            batch_first=True
        )

        # stack encoder layers to obtain Encoder
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_encoder_layers,
            norm=None
        )

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
            dropout=dropout,
            batch_first=True
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

    def pos_encoding(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, enc_seq_len, dim_val]

        returns: Tensor, shape [batch_size,enc_seq_len, dim_val]
        """
        #print(f"forward tensor shape: {x.shape}, pos enc shape: {self.pe[:, :x.size(1)].shape}")
        x = x + self.pe[:,:x.size(1)]

        return self.dropout(x)


    def forward(self, tgt: Tensor, src: Tensor, tgt_mask: Tensor=None, src_mask: Tensor=None) -> Tensor:

        src = self.encoder_input_layer(src)

        # Pass through the positional encoding layer
        # src = src.unsqueeze(0)
        pos_encoded_src = self.pos_encoding(src)

        encoder_output = self.encoder(
            src=pos_encoded_src
        )

        tgt = self.decoder_input_layer(tgt)

        decoder_output = self.decoder(
            tgt=tgt,
            memory=src,
            tgt_mask=tgt_mask,
            memory_mask=src_mask
        )

        decoder_output = self.linear_mapping(decoder_output)

        return decoder_output
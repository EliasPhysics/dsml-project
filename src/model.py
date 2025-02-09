from torch import nn, Tensor

from encoder import Encoder
from decoder import Decoder


class TimeSeriesTransformer(nn.Module):
    def __init__(self,
                 input_size: int,
                 d_model: int,
                 n_encoder_layers: int,
                 n_decoder_layers: int,
                 dec_seq_len: int,
                 dropout_encoder: float = 0.2,
                 dropout_pos_enc: float = 0.2,
                 max_seq_len: int = 5000,
                 dim_feedforward_encoder: int = 2048,
                 dropout_decoder: float = 0.2,
                 n_heads: int = 8,
                 dim_feedforward_decoder: int = 2048,
                 num_predicted_features: int = 1
                 ):
        super().__init__()

        self.encoder = Encoder(input_size=input_size,
                               d_model=d_model,
                               nhead=n_heads,
                               n_encoder_layers=n_encoder_layers,
                               dropout_encoder=dropout_encoder,
                               dropout_pos_enc=dropout_pos_enc,
                               max_seq_len=max_seq_len,
                               dim_feedforward_encoder=dim_feedforward_encoder
                               )

        self.decoder = Decoder(d_model=d_model,
                 n_decoder_layers=n_decoder_layers,
                 dec_seq_len=dec_seq_len,
                 dropout_decoder=dropout_decoder,
                 n_heads=n_heads,
                 dim_feedforward_decoder=dim_feedforward_decoder,
                 num_predicted_features=num_predicted_features
                 )

    def forward(self, tgt: Tensor, src: Tensor, tgt_mask: Tensor=None, src_mask: Tensor=None) -> Tensor:

        encoder_output = self.encoder(src=src)
        decoder_output = self.decoder(tgt=tgt, src=encoder_output,tgt_mask=tgt_mask,src_mask=src_mask)

        return decoder_output

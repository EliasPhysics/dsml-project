import torch
from torch.utils.data import Dataset
from typing import Tuple


class TransformerDataset(Dataset):
    def __init__(self,
                 data: torch.tensor,
                 indices: list,
                 enc_seq_len: int,
                 dec_seq_len: int,
                 target_seq_len: int
                 ) -> None:
        super().__init__()
        self.indices = indices
        self.data = data
        self.enc_seq_len = enc_seq_len
        self.dec_seq_len = dec_seq_len
        self.target_seq_len = target_seq_len

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        """
        Returns a tuple with 3 elements:
        1) src (the encoder input)
        2) trg (the decoder input)
        3) trg_y (the target)
        """
        # Get the first element of the i'th tuple in the list self.indicesasdfas
        start_idx = self.indices[index][0]

        # Get the second (and last) element of the i'th tuple in the list self.indices
        end_idx = self.indices[index][1]

        sequence = self.data[start_idx:end_idx]

        src, trg, trg_y = self.get_src_trg(
            sequence=sequence,
            enc_seq_len=self.enc_seq_len,
            dec_seq_len=self.dec_seq_len,
            target_seq_len=self.target_seq_len
        )

        return src, trg, trg_y

    def get_src_trg(
            self,
            sequence: torch.Tensor,
            enc_seq_len: int,
            dec_seq_len: int,
            target_seq_len: int
    ) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:

        assert len(
            sequence) == enc_seq_len + target_seq_len, "Sequence length does not equal (input length + target length)"

        # encoder input
        src = sequence[:enc_seq_len]
        trg = sequence[enc_seq_len - 1:len(sequence) - 1]
        assert len(trg) == target_seq_len, "Length of trg does not match target sequence length"

        # The target sequence against which the model output will be compared to compute loss
        trg_y = sequence[-target_seq_len:]
        assert len(trg_y) == target_seq_len, "Length of trg_y does not match target sequence length"

        return src, trg, trg_y
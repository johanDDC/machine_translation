import torch
from torch import nn
from torch import Tensor

from src.models.mask_predict.decoder.decoder import Decoder
from src.models.mask_predict.encoder.encoder import Encoder


class MaskPredictModel(nn.Module):
    def __init__(self, vocab_size, src_max_len, tgt_max_len, n_layers_enc,
                 n_layers_dec, d_model, d_inner, n_heads, device, dropout):
        super().__init__()
        self.encoder = Encoder(n_layers_enc, src_max_len, tgt_max_len, vocab_size,
                               d_model, d_inner, n_heads, dropout)
        self.decoder = Decoder(n_layers_dec, src_max_len, tgt_max_len, vocab_size,
                               d_model, d_inner, n_heads, dropout)

    def forward(self, src_seq, src_pos, src_mask, src_len,
                result_tokens=None):
        encoded_seq, len_preds, src_mask = self.encoder(src_seq, src_pos, src_mask)


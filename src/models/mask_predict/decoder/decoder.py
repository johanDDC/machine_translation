import torch
from torch import nn
from torch import Tensor

from src.models.mask_predict.decoder.fft_block import FFTBlock
from src.models.utils import SequenceEmbeddings


class Decoder(nn.Module):
    def __init__(self, n_layers, src_max_len, tgt_max_len, vocab_size, d_model, d_inner,
                 n_heads, dropout):
        super().__init__()
        self.n_layers = n_layers

        self.tgt_embeddings = SequenceEmbeddings(vocab_size, tgt_max_len, d_model)
        self.fft_layers = nn.ModuleList(
            [FFTBlock(d_model, d_inner, n_heads, dropout) for _ in range(n_layers)]
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, tgt_seq, tgt_pos, kv_encoder, enc_mask=None, tgt_mask=None):
        x = self.tgt_embeddings(tgt_seq, tgt_pos)

        for i in range(self.n_layers):
            x, attention_map_dec, attention_map_enc = self.fft_layers[i](x,
                                                                         kv_encoder,
                                                                         enc_mask,
                                                                         tgt_mask)

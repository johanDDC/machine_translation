import torch
from torch import nn
from torch import Tensor

from src.models.mask_predict.decoder.fft_block import FFTBlock
from src.models.model_utils import SequenceEmbeddings


class Decoder(nn.Module):
    def __init__(self, n_layers, src_max_len, tgt_max_len, vocab_size, d_model, d_inner,
                 n_heads, dropout):
        super().__init__()
        self.n_layers = n_layers

        self.tgt_embeddings = SequenceEmbeddings(vocab_size, tgt_max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.fft_layers = nn.ModuleList(
            [FFTBlock(d_model, d_inner, n_heads, dropout) for _ in range(n_layers)]
        )
        self.project_vocab = nn.Linear(d_model, vocab_size)

    def forward(self, tgt_input, tgt_pos, kv_encoder, enc_pad_mask=None, tgt_pad_mask=None):
        x = self.tgt_embeddings(tgt_input, tgt_pos)
        x = self.dropout(x)

        for i in range(self.n_layers):
            x, attention_map_dec, attention_map_enc = self.fft_layers[i](x,
                                                                         kv_encoder,
                                                                         enc_pad_mask,
                                                                         tgt_pad_mask)
        output = self.project_vocab(x)
        return output

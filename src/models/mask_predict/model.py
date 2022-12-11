import torch
from torch import nn
from torch import Tensor
from torch.nn import init

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

        self.apply(self.__init_weights)

    def __init_weights(self, layer):
        if isinstance(layer, (nn.Conv2d, nn.Linear, nn.Embedding)):
            init.normal_(layer.weight, mean=0, std=0.02)
            try:
                init.constant_(layer.bias, 0)
            except:
                pass
        if isinstance(layer, nn.LayerNorm):
            init.constant_(layer.weight, 1)
            init.constant_(layer.bias, 0)

    def forward(self, src_seq, src_pos, src_padding_mask,
                tgt_input, tgt_pos, tgt_padding_mask, **kwargs):
        encoded_seq, len_preds = self.encoder(src_seq, src_pos, src_padding_mask)
        decoder_output = self.decoder(tgt_input, tgt_pos, encoded_seq, src_padding_mask, tgt_padding_mask)
        return {
            "output": decoder_output,
            "len_prediction": len_preds
        }

    def encode(self, src: Tensor, src_pos: Tensor, src_pad_mask: Tensor):
        return self.encoder(src, src_pos, src_pad_mask)

    def decode(self, tgt: Tensor, tgt_pos:Tensor, memory: Tensor,
               memory_pad_mask:Tensor, tgt_pad_mask: Tensor):
        return self.decoder(tgt, tgt_pos, memory, memory_pad_mask, tgt_pad_mask)



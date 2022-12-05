import torch
from torch import nn


class FFTBlock(nn.Module):
    def __init__(self,
                 d_model,
                 d_inner,
                 n_head,
                 dropout=0.1,
                 attention_dropout=0.1):
        super(FFTBlock, self).__init__()
        self.slf_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_head,
                                              dropout=attention_dropout, bias=False, batch_first=True)
        self.encoder_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_head,
                                                  dropout=attention_dropout, bias=False, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_inner),
            nn.ReLU(),
            nn.Dropout(attention_dropout),
            nn.Linear(d_inner, d_model),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, kv_encoder, non_pad_mask_enc=None, non_pad_mask_dec=None):
        residual = x.clone()
        x = self.ln1(x)
        attention_output, attention_map_dec = self.slf_attn(x, x, x, key_padding_mask=non_pad_mask_dec)
        x = residual + self.dropout(attention_output)
        if non_pad_mask_dec is not None:
            x = x.masked_fill(non_pad_mask_dec.unsqueeze(-1), 0)

        residual = x.clone()
        x = self.ln2(x)
        attention_output, attention_map_enc = self.encoder_attn(query=x,
                                                                key=kv_encoder,
                                                                value=kv_encoder,
                                                                key_padding_mask=non_pad_mask_enc)
        x = residual + self.dropout(attention_output)
        if non_pad_mask_enc is not None:
            x = x.masked_fill(non_pad_mask_enc.unsqueeze(-1), 0)


        residual = x.clone()
        x = self.ln3(x)
        x = self.feed_forward(x)
        x = residual + self.dropout(x)
        if non_pad_mask_dec is not None:
            x = x.masked_fill(non_pad_mask_dec.unsqueeze(-1), 0)

        return x, attention_map_dec, attention_map_enc

import torch
from torch import nn


class FFTBlock(nn.Module):
    def __init__(self,
                 d_model,
                 d_inner,
                 n_head,
                 dropout=0.1):
        super(FFTBlock, self).__init__()
        self.slf_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_head,
                                              dropout=dropout, bias=False, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_inner),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, non_pad_mask=None):
        residual = x.clone()
        x = self.ln1(x)
        attention_output, attention_map = self.slf_attn(x, x, x, key_padding_mask=non_pad_mask)
        x = residual + self.dropout(attention_output)

        residual = x.clone()
        x = self.ln2(x)
        x = self.feed_forward(x)
        x = residual + self.dropout(x)

        return x, attention_map

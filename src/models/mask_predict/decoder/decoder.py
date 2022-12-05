import torch
from torch import nn
from torch import Tensor

class Decoder(nn.Module):
    def __init__(self, n_layers, src_max_len, tgt_max_len, vocab_size, d_model, d_inner,
                 n_heads, dropout):
        super().__init__()
        self.n_layers = n_layers

        self.src_embeddings = SequenceEmbeddings(vocab_size, src_max_len, d_model)
        self.len_embeddings = nn.Embedding(tgt_max_len, d_model)
        self.fft_layers = nn.ModuleList(
            [FFTBlock(d_model, d_inner, n_heads, dropout) for _ in range(n_layers)]
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, src_seq, src_pos, src_mask=None):
        x = self.src_embeddings(src_seq, src_pos) # B x Ls x D
        LEN_tokens = torch.zeros((x.shape[0], 1), device=x.device) # B x 1
        LEN_tokens = self.len_embeddings(LEN_tokens) # B x D
        x = torch.cat([LEN_tokens, x], dim=1) # B x (Ls + 1) x D

        src_mask = torch.cat([torch.zeros((x.shape[0], 1), device=x.device),
                              src_mask], dim=1) # B x (Ls + 1)


        for i in range(self.n_layers):
            x, _ = self.fft_layers[i](x, src_mask)

        LEN_predicted = x[:, 0, :] @ self.len_embeddings.weight.T # B x Lt
        LEN_predicted[:, 0] = -torch.inf
        LEN_predicted = self.softmax(LEN_predicted)

        return x, LEN_predicted, src_mask


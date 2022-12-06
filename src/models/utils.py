import torch.nn as nn
from torch import Tensor

class SequenceEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, n_positions: int, emb_size: int):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, emb_size)
        self.pos_embeddings = nn.Embedding(n_positions, emb_size, padding_idx=0)

    def forward(self, seq: Tensor, seq_pos: Tensor):
        return self.embeddings(seq) + self.pos_embeddings(seq_pos)

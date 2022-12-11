import torch.nn as nn
from torch import Tensor
from torch.nn.init import xavier_normal_, constant_

from src.models.model_utils import SequenceEmbeddings


class TranslationModel(nn.Module):
    def __init__(
            self,
            num_encoder_layers: int,
            num_decoder_layers: int,
            emb_size: int,
            dim_feedforward: int,
            n_head: int,
            src_vocab_size: int,
            tgt_vocab_size: int,
            dropout_prob: float,
            n_positions: int
    ):
        """
        Creates a standard Transformer encoder-decoder model.
        :param num_encoder_layers: Number of encoder layers
        :param num_decoder_layers: Number of decoder layers
        :param emb_size: Size of intermediate vector representations for each token
        :param dim_feedforward: Size of intermediate representations in FFN layers
        :param n_head: Number of attention heads for each layer
        :param src_vocab_size: Number of tokens in the source language vocabulary
        :param tgt_vocab_size: Number of tokens in the target language vocabulary
        :param dropout_prob: Dropout probability throughout the model
        """
        super().__init__()
        self.src_embeddings = SequenceEmbeddings(src_vocab_size, n_positions, emb_size)
        self.tgt_embeddings = SequenceEmbeddings(src_vocab_size, n_positions, emb_size)
        self.transformer = nn.Transformer(d_model=emb_size, nhead=n_head, num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward,
                                          dropout=dropout_prob, batch_first=True, norm_first=True)
        self.head = nn.Linear(emb_size, tgt_vocab_size)

        self.apply(self.__init_weights)

    def __init_weights(self, layer):
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            xavier_normal_(layer.weight)
            constant_(layer.bias, 0)

    def forward(
            self,
            src_seq: Tensor,
            src_pos: Tensor,
            tgt_seq: Tensor,
            tgt_pos: Tensor,
            tgt_mask: Tensor,
            src_padding_mask: Tensor,
            tgt_padding_mask: Tensor,
            **kwargs
    ):
        """
        Given tokens from a batch of source and target sentences, predict logits for next tokens in target sentences.
        """
        local_tgt_seq = tgt_seq[:, :-1]
        local_tgt_pos = tgt_pos[:, :-1]
        tgt_len = local_tgt_seq.shape[1]
        local_tgt_padding = tgt_padding_mask[:, :tgt_len]
        local_tgt_mask = tgt_mask[:tgt_len, :tgt_len]

        src_embeded = self.src_embeddings(src_seq, src_pos)
        tgt_embeded = self.tgt_embeddings(local_tgt_seq, local_tgt_pos)
        output = self.transformer(src=src_embeded, tgt=tgt_embeded, src_mask=None, tgt_mask=local_tgt_mask,
                                  memory_mask=None, src_key_padding_mask=src_padding_mask,
                                  tgt_key_padding_mask=local_tgt_padding, memory_key_padding_mask=None)
        return self.head(output)

    def encode(self, src: Tensor, src_pos:Tensor, src_padding_mask: Tensor):
        src_embeded = self.src_embeddings(src, src_pos)
        return self.transformer.encoder(src=src_embeded, src_key_padding_mask=src_padding_mask)

    def decode(self, tgt: Tensor, tgt_pos:Tensor, memory: Tensor, tgt_mask: Tensor):
        tgt_embeded = self.tgt_embeddings(tgt, tgt_pos)
        return self.transformer.decoder(tgt=tgt_embeded, memory=memory, tgt_mask=tgt_mask)

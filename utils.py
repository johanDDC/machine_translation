import torch.nn as nn
import torch
import random
import numpy as np
from sacremoses import MosesDetokenizer, MosesPunctNormalizer, MosesTokenizer

from src.data.data import SpecialTokens, TranslationDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cpu")


class Dict(dict):
    def __init__(self, dct=None):
        super().__init__()
        dct = dict() if not dct else dct
        for key, value in dct.items():
            if hasattr(value, "keys"):
                value = Dict(value)
            self[key] = value

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def generate_square_subsequent_mask(sz, device="cpu"):
    mask = -torch.inf * torch.ones((sz, sz), device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask


def create_mask(src, tgt, device="cpu", PAD_IDX=0):
    tgt_seq_len = tgt.shape[1]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device)

    src_padding_mask = (src == PAD_IDX)
    tgt_padding_mask = (tgt == PAD_IDX)
    return tgt_mask, src_padding_mask, tgt_padding_mask


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
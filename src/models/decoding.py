import torch
from sacremoses import MosesDetokenizer, MosesPunctNormalizer, MosesTokenizer
from tokenizers import Tokenizer
from typing import List

from tqdm import tqdm

from src.models.vaswani import TranslationModel

# it's a surprise tool that will help you later
from src.data.data import SpecialTokens
from utils import generate_square_subsequent_mask

detok = MosesDetokenizer(lang="en")
mpn = MosesPunctNormalizer()


def _greedy_decode(
        model: TranslationModel,
        src: torch.Tensor,
        src_pos: torch.Tensor,
        max_len: int,
        tgt_tokenizer: Tokenizer,
        device: torch.device,
) -> torch.Tensor:
    """
    Given a batch of source sequences, predict its translations with greedy search.
    The decoding procedure terminates once either max_len steps have passed
    or the "end of sequence" token has been reached for all sentences in the batch.
    :param model: the model to use for translation
    :param src: a (batch, time) tensor of source sentence tokens
    :param max_len: the maximum length of predictions
    :param tgt_tokenizer: target language tokenizer
    :param device: device that the model runs on
    :return: a (batch, time) tensor with predictions
    """
    EOS_IDX = tgt_tokenizer.token_to_id(SpecialTokens.END.value)
    memory = model.encode(src, src_pos)
    y = torch.tensor(src[0], dtype=torch.long, device=device).view(1)
    y_pos = []
    for i in range(max_len - 1):
        y_mask = generate_square_subsequent_mask(y.shape[0], device=device).bool()
        y_pos.append(i + 1)
        out = model.decode(y, torch.tensor(y_pos, dtype=torch.long, device=device).view(y.shape), memory, y_mask)
        probs = model.head(out[-1])
        next_word = probs.argmax().item()
        if next_word == EOS_IDX:
            break
        y = torch.cat([y,
                       torch.tensor(next_word, dtype=torch.long, device=device).view(1)])
    return y


def _beam_search_decode(
        model: TranslationModel,
        src: torch.Tensor,
        src_pos: torch.Tensor,
        max_len: int,
        tgt_tokenizer: Tokenizer,
        device: torch.device,
        beam_size: int,
) -> torch.Tensor:
    """
    Given a batch of source sequences, predict its translations with beam search.
    The decoding procedure terminates once max_len steps have passed.
    :param model: the model to use for translation
    :param src: a (batch, time) tensor of source sentence tokens
    :param max_len: the maximum length of predictions
    :param tgt_tokenizer: target language tokenizer
    :param device: device that the model runs on
    :param beam_size: the number of hypotheses
    :return: a (batch, time) tensor with predictions
    """
    pass


@torch.inference_mode()
def translate(
        model: TranslationModel,
        src_sentences: List[str],
        src_tokenizer: Tokenizer,
        tgt_tokenizer: Tokenizer,
        translation_mode: str,
        device: torch.device,
) -> List[str]:
    """
    Given a list of sentences, generate their translations.
    :param model: the model to use for translation
    :param src_sentences: untokenized source sentences
    :param src_tokenizer: source language tokenizer
    :param tgt_tokenizer: target language tokenizer
    :param translation_mode: either "greedy", "beam" or anything more advanced
    :param device: device that the model runs on
    """
    model.eval()
    res = []
    detokenizer = MosesDetokenizer(lang="en")
    tokenizer = MosesTokenizer(lang="en")
    normalizer = MosesPunctNormalizer()
    if translation_mode == "greedy":
        method = _greedy_decode
    else:
        method = _beam_search_decode
    BOS_idx = src_tokenizer.token_to_id(SpecialTokens.BEGINNING.value)
    EOS_idx = src_tokenizer.token_to_id(SpecialTokens.END.value)
    for sentence in tqdm(src_sentences):
        tokens = src_tokenizer.encode(sentence.strip("\n")).ids
        tokens = torch.tensor([BOS_idx] + tokens + [EOS_idx], device=device)
        src_pos = torch.arange(1, len(tokens) + 1, device=device)
        tokens_pred = method(model, tokens, src_pos, int(1.5 * len(tokens)), tgt_tokenizer, device)
        tokens_pred = tgt_tokenizer.decode(tokens_pred.tolist())
        tokens_pred = tokenizer.tokenize(tokens_pred)
        tokens_pred = detokenizer.detokenize(tokens_pred)
        tokens_pred = normalizer.normalize(tokens_pred)
        res.append(tokens_pred)
    return res
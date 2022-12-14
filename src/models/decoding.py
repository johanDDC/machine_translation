import torch
import torch.nn.functional as F
from sacremoses import MosesDetokenizer, MosesPunctNormalizer, MosesTokenizer
from tokenizers import Tokenizer
from typing import List

from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

from src.models.mask_predict.model import MaskPredictModel
from src.models.vaswani.model import TranslationModel

# it's a surprise tool that will help you later
from src.data.data import SpecialTokens
from utils import generate_square_subsequent_mask

detok = MosesDetokenizer(lang="en")
mpn = MosesPunctNormalizer()


def _batch_greedy_decode(
        model: TranslationModel,
        src: torch.Tensor,
        src_pos: torch.Tensor,
        src_pad_mask,
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
    memory = model.encode(src, src_pos, src_pad_mask)
    y = torch.full((src.shape[0], 1), src[0, 0], dtype=torch.long, device=device)
    y_pos = torch.ones_like(y, dtype=torch.long, device=device)
    for i in range(1, max_len):
        y_mask = generate_square_subsequent_mask(y.shape[1], device=device).bool()
        out = model.decode(y, y_pos, memory, y_mask)
        probs = model.head(out[:, -1, :])
        next_tokens = probs.argmax(1).view(-1, 1)
        y = torch.cat([y, next_tokens], dim=1)
    return y


def _greedy_decode(
        model: TranslationModel,
        src: torch.Tensor,
        src_pos: torch.Tensor,
        src_pad_mask,
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
    PAD_idx = tgt_tokenizer.token_to_id(SpecialTokens.PADDING.value)
    res = torch.full((src.shape[0], max_len), PAD_idx, dtype=torch.long, device=device)
    for i in range(src.shape[0]):
        memory = model.encode(src[i], src_pos[i], src_pad_mask[i])
        y = torch.tensor(src[0, 0].item(), dtype=torch.long, device=device).view(1)
        y_pos = []
        for j in range(max_len - 1):
            y_mask = generate_square_subsequent_mask(y.shape[0], device=device).bool()
            y_pos.append(j + 1)
            out = model.decode(y, torch.tensor(y_pos, dtype=torch.long, device=device).view(y.shape), memory, y_mask)
            probs = model.head(out[-1])
            next_word = probs.argmax().item()
            y = torch.cat([y, torch.tensor(next_word, dtype=torch.long, device=device).view(1)])
            if next_word == EOS_IDX:
                break

        res[i, :len(y)] = y
    return res



def _mask_predict_decode(
        model: MaskPredictModel,
        src: torch.Tensor,
        src_pos: torch.Tensor,
        src_pad_mask: torch.Tensor,
        max_len: int,
        tgt_tokenizer: Tokenizer,
        device: torch.device,
        len_candidates=3,
        num_iters=3):

    def mask(cur_iter, max_iters, lens, pred_tokens=None, pred_probas=None):
        MASK_IDX = tgt_tokenizer.token_to_id(SpecialTokens.MASK.value)
        PAD_idx = tgt_tokenizer.token_to_id(SpecialTokens.PADDING.value)
        if cur_iter == 0:
            y = torch.full((src.shape[0], max_len), fill_value=PAD_idx)
            y_pos = torch.zeros((src.shape[0], max_len))
            mask_ids = []
            for i in range(src.shape[0]):
                y[i, :lens[i]] = MASK_IDX
                mask_ids.append(torch.arange(1, lens[i] + 1))
                y_pos[i, :lens[i]] = mask_ids[-1]
            y = y.to(device, non_blocking=True)
            y_pos = y_pos.to(device, dtype=torch.long, non_blocking=True)
        else:
            mask_count = (lens * ((max_iters - cur_iter) / max_iters)).long()
            mask_ids = []
            y = pad_sequence(pred_tokens, batch_first=True, padding_value=PAD_idx)
            y_pos = torch.zeros_like(y)
            # torch.scatter_ would be more effective
            for i in range(src.shape[0]):
                mask_ids_ = torch.argsort(pred_probas[i])[:mask_count[i]]
                mask_ids.append(mask_ids_)
                y[i, mask_ids[i]] = MASK_IDX
                y_pos[i, :len(pred_tokens[i])] = torch.arange(1, len(pred_tokens[i]) + 1)
            y_pos = y_pos.to(device, dtype=torch.long, non_blocking=True)

        return y, y_pos, mask_ids

    def predict(y: torch.Tensor, y_pos: torch.Tensor, memory: torch.Tensor, y_pad_mask):
        # memory ?????????? ?????????? ?????????????? ????????????
        token_logits = model.decode(y, y_pos, memory, src_pad_mask, y_pad_mask)
        token_probas = F.log_softmax(token_logits, -1)
        pred_probas, pred_tokens = torch.max(token_probas, dim=-1)
        return pred_tokens, pred_probas

    memory, len_preds = model.encode(src, src_pos, src_pad_mask)
    len_preds = len_preds.argmax(-1)
    pred_tokens = None
    pred_probas = None
    y_pad_mask = None
    PAD_idx = tgt_tokenizer.token_to_id(SpecialTokens.PADDING.value)
    for it in range(num_iters):
        y, y_pos, mask_ids = mask(it, num_iters, len_preds, pred_tokens, pred_probas)
        y_pad_mask = (y == PAD_idx)
        pred_tokens_new, pred_probas_new = predict(y, y_pos, memory, y_pad_mask)
        if pred_tokens is None:
            pred_tokens = [pred_tokens_new[i, mask_ids[i]] for i in range(src.shape[0])]
            pred_probas = [pred_probas_new[i, mask_ids[i]] for i in range(src.shape[0])]
        else:
            for i in range(src.shape[0]):
                pred_tokens[i][mask_ids[i]] = pred_tokens_new[i, mask_ids[i]]

    return pred_tokens

def _mask_predict_decode_v2(model: MaskPredictModel,
        src: torch.Tensor,
        src_pos: torch.Tensor,
        src_pad_mask: torch.Tensor,
        max_len: int,
        tgt_tokenizer: Tokenizer,
        device: torch.device,
        len_candidates=3,
        num_iters=3):
    result = []
    for i in range(src.shape[0]):
        src_ = src[i]
        src_ = src_.masked_select(~src_pad_mask[i]).view(1, -1).to(device)
        src_pos_ = torch.arange(1, src_.shape[1] + 1).to(device).view(1, -1)
        src_pad = torch.zeros_like(src_pos_, device=device).view(1, -1).bool()
        out, lng = model.encode(src_, src_pos_, src_pad)
        len_candidates_ = []
        lengthes = torch.argsort(lng)[0, -len_candidates:]
        for lng in lengthes:
            y = 4 * torch.ones((1, lng), device=device).view(1, -1).long()
            y_pos = torch.arange(1, lng + 1, device=device).view(1, -1).long()
            y_pad = torch.zeros_like(y_pos, device=device).view(1, -1).bool()
            y_proba = torch.zeros_like(y, device=device).view(1, -1).float()
            masking_ids = torch.arange(0, lng, device=device)
            for it in range(num_iters):
                y[0, masking_ids] = 4
                res = model.decode(y, y_pos, out, src_pad, y_pad)
                probas = F.log_softmax(res, -1).squeeze()
                prb, idx = torch.max(probas, -1)
                y[0, masking_ids] = idx[masking_ids]
                y_proba[0, masking_ids] = prb[masking_ids]
                masking_ids = torch.argsort(prb)[:int(lng * (num_iters - it) / num_iters)]
            len_candidates_.append((y[0], y_proba.mean()))
        result.append(sorted(len_candidates_, key=lambda x: x[1])[-1][0])
    return result


def _batch_beam_search_decode(
        model: TranslationModel,
        src: torch.Tensor,
        src_pos: torch.Tensor,
        src_pad_mask,
        max_len: int,
        tgt_tokenizer: Tokenizer,
        device: torch.device,
        beam_size=5,
):
    EOS_idx = tgt_tokenizer.token_to_id(SpecialTokens.END.value)
    PAD_idx = tgt_tokenizer.token_to_id(SpecialTokens.PADDING.value)
    BOS_idx = tgt_tokenizer.token_to_id(SpecialTokens.BEGINNING.value)
    batch_size = src.shape[0]
    res = torch.zeros((src.shape[0], max_len), device=device)

    memory = model.encode(src, src_pos, src_pad_mask)
    temp_ids = torch.arange(batch_size, device=device).view(-1, 1).repeat(1, beam_size * 2).view(-1)
    memory = memory.index_select(0, temp_ids)
    seq_score = torch.zeros((batch_size * beam_size * 2, max_len), dtype=torch.float32, device=device)
    seq_tokens = torch.full((batch_size * beam_size * 2, max_len), PAD_idx, dtype=torch.long, device=device)
    seq_tokens[:, 0] = BOS_idx
    seq_pos = torch.arange(1, max_len + 1, dtype=torch.long, device=device).view(-1, 1).repeat(1, batch_size * beam_size * 2).T
    finalized = torch.full((batch_size * beam_size * 2,), -torch.inf, dtype=torch.float32, device=device)
    for j in range(1, max_len):
        y = seq_tokens[:, :j]
        y_pos = seq_pos[:, :j]
        y_mask = generate_square_subsequent_mask(y.shape[1], device=device).bool()
        out = model.decode(y, y_pos, memory, y_mask)
        log_probs = F.log_softmax(model.head(out[:, -1, :]), -1)
        log_probs = log_probs.view(batch_size, beam_size * 2, -1)
        vals, ids = torch.topk(log_probs, k=beam_size * 2, dim=-1)
        if j == 1:
            seq_score[:, j] = vals[:, 0, :].reshape(-1)
            seq_tokens[:, j] = ids[:, 0, :].reshape(-1)
        else:
            col_accumulated_lprobs = seq_score.sum(1).view(batch_size, -1, 1)
            accumulated_lprobs_with_next = col_accumulated_lprobs + vals
            best_nexts, best_ids = torch.topk(accumulated_lprobs_with_next.view(batch_size, -1), k=beam_size * 2, dim=-1)
            seq_score[:, j] = best_nexts.view(-1)
            best_ids = ids.view(batch_size, -1).gather(1, best_ids)
            seq_tokens[:, j] = best_ids.view(-1)
            if len(torch.where(best_ids == EOS_idx)[0]) > 0:
                where_eos = torch.where(best_ids == EOS_idx)[0]
                finalized[where_eos] = torch.maximum(seq_score[where_eos].sum(1), finalized[where_eos])

    res_seq_scores = seq_score.sum(1).view(-1)
    res_seq_scores[finalized > -torch.inf] = finalized[finalized > -torch.inf]
    best_scores = res_seq_scores.view(batch_size, -1).argmax(-1).view(batch_size, 1)
    res_tokens = seq_tokens.view(batch_size, beam_size * 2, -1)
    res_tokens = res_tokens.gather(1, best_scores.repeat(1, res_tokens.shape[-1]).view(batch_size, 1, -1))
    return res_tokens.squeeze(1)


def _beam_search_decode(
        model: TranslationModel,
        src: torch.Tensor,
        src_pos: torch.Tensor,
        src_pad_mask,
        max_len: int,
        tgt_tokenizer: Tokenizer,
        device: torch.device,
        beam_size=5,
):
    EOS_idx = tgt_tokenizer.token_to_id(SpecialTokens.END.value)
    PAD_idx = tgt_tokenizer.token_to_id(SpecialTokens.PADDING.value)
    BOS_idx = tgt_tokenizer.token_to_id(SpecialTokens.BEGINNING.value)
    res = torch.zeros((src.shape[0], max_len), device=device)
    for i in range(src.shape[0]):
        memory = model.encode(src[i], src_pos[i], src_pad_mask[i])
        temp_ids = torch.arange(1, device=device).view(-1, 1).repeat(1, beam_size * 2).view(-1)
        memory = memory.unsqueeze(0).index_select(0, temp_ids)
        seq_score = torch.zeros((beam_size * 2, max_len), dtype=torch.float32, device=device)
        seq_tokens = torch.full((beam_size * 2, max_len), PAD_idx, dtype=torch.long, device=device)
        seq_tokens[:, 0] = BOS_idx
        seq_pos = torch.arange(1, max_len + 1, dtype=torch.long, device=device).view(-1, 1).repeat(1, beam_size * 2).T
        finalized = torch.full((beam_size * 2,), -torch.inf, dtype=torch.float32, device=device)
        for j in range(1, max_len):
            y = seq_tokens[:, :j]
            y_pos = seq_pos[:, :j]
            y_mask = generate_square_subsequent_mask(y.shape[1], device=device).bool()
            out = model.decode(y, y_pos, memory, y_mask)
            log_probs = F.log_softmax(model.head(out[:, -1, :]), -1)
            vals, ids = torch.topk(log_probs, beam_size * 2)
            if j == 1:
                seq_score[:, j] = vals[0]
                seq_tokens[:, j] = ids[0]
            else:
                col_accumulated_lprobs = seq_score.sum(1).view(-1, 1)
                accumulated_lprobs_with_next = col_accumulated_lprobs + vals
                accumulated_lprobs_with_next = accumulated_lprobs_with_next.view(-1)
                best_nexts, best_ids = torch.topk(accumulated_lprobs_with_next, k=beam_size * 2)
                seq_score[:, j] = best_nexts
                best_ids = ids.view(-1)[best_ids]
                seq_tokens[:, j] = best_ids
                if len(torch.where(best_ids == EOS_idx)[0]) > 0:
                    where_eos = torch.where(best_ids == EOS_idx)[0]
                    finalized[where_eos] = torch.maximum(seq_score[where_eos].sum(1), finalized[where_eos])
                    if len(finalized[finalized > -torch.inf]) >= beam_size:
                        break
        res_seq_scores = seq_score.sum(1)
        res_seq_scores[finalized > -torch.inf] = finalized[[finalized > -torch.inf]]
        res[i] = seq_tokens[res_seq_scores.argmax()]
    return res


class __TestDataset(torch.utils.data.Dataset):
    def __init__(self, src_sentences: List[str],
                 src_tokenizer: Tokenizer,
                 tgt_tokenizer: Tokenizer,
                 translation_mode: str):
        self.sentences = src_sentences
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.translation_mode = translation_mode

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, item):
        return self.sentences[item]

    def collator(self, batch):
        BOS_idx = self.src_tokenizer.token_to_id(SpecialTokens.BEGINNING.value)
        EOS_idx = self.src_tokenizer.token_to_id(SpecialTokens.END.value)
        PAD_idx = self.src_tokenizer.token_to_id(SpecialTokens.PADDING.value)
        src_pos = []
        src_batch = []
        for sentence in batch:
            tokens = self.src_tokenizer.encode(sentence.strip("\n")).ids
            if self.translation_mode != "mask_predict":
                tokens = torch.tensor([BOS_idx] + tokens + [EOS_idx])
            else:
                tokens = torch.tensor(tokens)

            src_batch.append(tokens)
            src_pos.append(torch.arange(1, len(tokens) + 1))

        src_pos = pad_sequence(src_pos, batch_first=True, padding_value=0)
        src_batch = pad_sequence(src_batch, batch_first=True, padding_value=PAD_idx)
        return {"src_seq": src_batch, "src_pos": src_pos}


def __translate_postprocess(tokens_pred, tgt_tokenizer):
    res = []
    detokenizer = MosesDetokenizer(lang="en")
    tokenizer = MosesTokenizer(lang="en")
    normalizer = MosesPunctNormalizer()
    EOS_idx = tgt_tokenizer.token_to_id(SpecialTokens.END.value)
    for prediction in tokens_pred:
        where_eos = torch.where(prediction == EOS_idx)[0]
        where_eos = where_eos[0].item() if len(where_eos) != 0 else prediction.shape[0]
        prediction = prediction[:where_eos].long()
        res_prediction = tgt_tokenizer.decode(prediction.tolist())
        res_prediction = tokenizer.tokenize(res_prediction)
        res_prediction = detokenizer.detokenize(res_prediction)
        res_prediction = normalizer.normalize(res_prediction)
        res.append(res_prediction)
    return res


class __TestDataset(torch.utils.data.Dataset):
    def __init__(self, src_sentences: List[str],
                 src_tokenizer: Tokenizer,
                 tgt_tokenizer: Tokenizer,
                 translation_mode: str):
        self.sentences = src_sentences
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.translation_mode = translation_mode

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, item):
        return self.sentences[item]

    def collator(self, batch):
        BOS_idx = self.src_tokenizer.token_to_id(SpecialTokens.BEGINNING.value)
        EOS_idx = self.src_tokenizer.token_to_id(SpecialTokens.END.value)
        PAD_idx = self.src_tokenizer.token_to_id(SpecialTokens.PADDING.value)
        src_pos = []
        src_batch = []
        for sentence in batch:
            tokens = self.src_tokenizer.encode(sentence.strip("\n")).ids
            if self.translation_mode != "mask_predict":
                tokens = torch.tensor([BOS_idx] + tokens + [EOS_idx])
            else:
                tokens = torch.tensor(tokens)

            src_batch.append(tokens)
            src_pos.append(torch.arange(1, len(tokens) + 1))

        src_pos = pad_sequence(src_pos, batch_first=True, padding_value=0)
        src_batch = pad_sequence(src_batch, batch_first=True, padding_value=PAD_idx)
        return {"src_seq": src_batch, "src_pos": src_pos}


def __translate_postprocess(tokens_pred, tgt_tokenizer):
    res = []
    detokenizer = MosesDetokenizer(lang="en")
    tokenizer = MosesTokenizer(lang="en")
    normalizer = MosesPunctNormalizer()
    for prediction in tokens_pred:
        res_prediction = tgt_tokenizer.decode(prediction.tolist())
        res_prediction = tokenizer.tokenize(res_prediction)
        res_prediction = detokenizer.detokenize(res_prediction)
        res_prediction = normalizer.normalize(res_prediction)
        res.append(res_prediction)
    return res


@torch.inference_mode()
def translate(
        model: TranslationModel,
        src_sentences: List[str],
        src_tokenizer: Tokenizer,
        tgt_tokenizer: Tokenizer,
        translation_mode: str,
        device: torch.device,
        data_cfg
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

    test_dataset = __TestDataset(src_sentences, src_tokenizer, tgt_tokenizer, translation_mode)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=data_cfg.batch_size_val,
                                                  shuffle=False,
                                                  num_workers=data_cfg.dataloader_num_workers,
                                                  pin_memory=True,
                                                  collate_fn=test_dataset.collator)

    if translation_mode == "greedy":
        method = _greedy_decode
    elif translation_mode == "beam":
        method = _beam_search_decode
    elif translation_mode == "mask_predict":
        method = _mask_predict_decode_v2
    elif translation_mode == "beam_batched":
        method = _batch_beam_search_decode
    res = []
    PAD_idx = src_tokenizer.token_to_id(SpecialTokens.PADDING.value)
    for batch in tqdm(test_dataloader):
        device_batch = {}
        for key in batch.keys():
            device_batch[key] = batch[key].to(device=device, dtype=torch.long,
                                              non_blocking=True, copy=False)
        src_pad_mask = (device_batch["src_seq"] == PAD_idx)
        tokens_pred = method(model, device_batch["src_seq"],
                             device_batch["src_pos"], src_pad_mask,  data_cfg.max_len,
                             tgt_tokenizer, device)

        res.extend(__translate_postprocess(tokens_pred, tgt_tokenizer))
    return res

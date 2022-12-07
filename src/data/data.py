import os
import time
from enum import Enum
from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


def process_training_file(input_path: Path, output_path: Path):
    """
    Processes raw training files ("train.tags.SRC-TGT.*"), saving the output as a sequence of unformatted examples
    (.txt file, one example per line).
    :param input_path: Path to the file with the input data (formatted examples)
    :param output_path: Path to the file with the output data (one example per line)
    """
    in_body = False
    description_tag = "<description>"
    description_stop = "TED Talk Subtitles and Transcript:"
    out_lines = []
    with open(input_path, "r") as f:
        for line in f:
            if line.find(description_tag) >= 0:
                description = line[line.find(description_tag) + len(description_tag)]
                description = description[len(description_stop):-len("</description>")]
                out_lines.append(description.strip())
                in_body = True
                continue
            if in_body and line.strip()[0] != "<":
                out_lines.append("".join([line.strip(), "\n"]))
            else:
                in_body = False
    with open(output_path, "w") as f:
        f.writelines(out_lines)
        f.flush()


def process_evaluation_file(input_path: Path, output_path: Path):
    """
    Processes raw validation and testing files ("IWSLT17.TED.{dev,test}2010.SRC-TGT.*.xml"),
    saving the output as a sequence of unformatted examples (.txt file, one example per line).
    :param input_path: Path to the file with the input data (formatted examples)
    :param output_path: Path to the file with the output data (one example per line)
    """
    seg = "<seg id=\""
    out_lines = []
    with open(input_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.find(seg) >= 0:
                subline = line[line.find(seg) + len(seg):-len("</seg>")]
                subline = subline[subline.find("\">") + 2:]
                out_lines.append("".join([subline.strip(), "\n"]))

    with open(output_path, "w") as f:
        f.writelines(out_lines)
        f.flush()


def convert_files(base_path: Path, output_path: Path):
    """
    Given a directory containing all the dataset files, convert each one into the "one example per line" format.
    :param base_path: Path containing files with original data
    :param output_path: Path containing files with processed data
    """
    os.makedirs(output_path, exist_ok=True)

    for language in "de", "en":
        process_training_file(
            base_path / f"train.tags.de-en.{language}",
            output_path / f"train.{language}.txt",
        )
        process_evaluation_file(
            base_path / f"IWSLT17.TED.dev2010.de-en.{language}.xml",
            output_path / f"val.{language}.txt",
        )
        process_evaluation_file(
            base_path / f"IWSLT17.TED.tst2010.de-en.{language}.xml",
            output_path / f"test.{language}.txt",
        )


class TranslationDataset(Dataset):
    def __init__(
            self,
            src_file_path,
            tgt_file_path,
            src_tokenizer: Tokenizer,
            tgt_tokenizer: Tokenizer,
            max_len=32,
    ):
        """
        Loads the training dataset and parses it into separate tokenized training examples.
        No padding should be applied at this stage
        :param src_file_path: Path to the source language training data
        :param tgt_file_path: Path to the target language training data
        :param src_tokenizer: Trained tokenizer for the source language
        :param tgt_tokenizer: Trained tokenizer for the target language
        :param max_len: Maximum length of source and target sentences for each example:
        if either of the parts contains more tokens, it needs to be filtered.
        """
        self.src = []
        self.tgt = []
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        for path, arr, tokenizer in zip([src_file_path, tgt_file_path],
                                        (self.src, self.tgt), (src_tokenizer, tgt_tokenizer)):
            with open(path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    arr.append(tokenizer.encode(line[:max_len].strip("\n")))

    def __len__(self):
        return len(self.src)

    def __getitem__(self, i):
        return (self.src[i].ids, self.tgt[i].ids)

    def collate_translation_data(self, batch):
        """
        Given a batch of examples with varying length, collate it into `source` and `target` tensors for the model.
        This method is meant to be used when instantiating the DataLoader class for training and validation datasets in your pipeline.
        """
        BOS_idx = self.src_tokenizer.token_to_id(SpecialTokens.BEGINNING.value)
        EOS_idx = self.src_tokenizer.token_to_id(SpecialTokens.END.value)
        PAD_idx = self.src_tokenizer.token_to_id(SpecialTokens.PADDING.value)
        src_batch, tgt_batch = [], []
        src_pos = []
        tgt_pos = []

        for src, tgt in batch:
            src = torch.tensor([BOS_idx] + src + [EOS_idx])
            tgt = torch.tensor([BOS_idx] + tgt + [EOS_idx])

            src_pos.append(torch.arange(1, len(src) + 1))
            tgt_pos.append(torch.arange(1, len(tgt) + 1))

            src_batch.append(src)
            tgt_batch.append(tgt)

        src_pos = pad_sequence(src_pos, batch_first=True, padding_value=0)
        tgt_pos = pad_sequence(tgt_pos, batch_first=True, padding_value=0)
        src_batch = pad_sequence(src_batch, batch_first=True, padding_value=PAD_idx)
        tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=PAD_idx)
        return {"src_seq": src_batch, "src_pos": src_pos,
                "tgt_seq": tgt_batch, "tgt_pos": tgt_pos}


class SpecialTokens(Enum):
    UNKNOWN = "[UNK]"
    PADDING = "[PAD]"
    BEGINNING = "[BOS]"
    END = "[EOS]"
    MASK = "[MASK]"


def train_tokenizers(base_dir: Path, save_dir: Path):
    """
    Trains tokenizers for source and target languages and saves them to `save_dir`.
    :param base_dir: Directory containing processed training and validation data (.txt files from `convert_files`)
    :param save_dir: Directory for storing trained tokenizer data (two files: `tokenizer_de.json` and `tokenizer_en.json`)
    """
    os.makedirs(save_dir, exist_ok=True)
    for language in ["en", "de"]:
        tokenizer = Tokenizer(BPE(unk_token=SpecialTokens.UNKNOWN.value))
        trainer = BpeTrainer(vocab_size=30_000, special_tokens=[e.value for e in SpecialTokens])
        tokenizer.pre_tokenizer = Whitespace()
        corpus = []
        with open(os.path.join(base_dir, f"train.{language}.txt"), "r") as f:
            corpus.extend(f.readlines())
        with open(os.path.join(base_dir, f"val.{language}.txt"), "r") as f:
            corpus.extend(f.readlines())
        tokenizer.train_from_iterator(corpus, trainer)
        tokenizer.save(os.path.join(save_dir, f"tokenizer_{language}.json"))

def mask_predict_collator(tokenizer):
    def collate(batch):
        PAD_idx = tokenizer.token_to_id(SpecialTokens.PADDING.value)
        src_batch, tgt_batch = [], []
        src_pos = []
        tgt_pos = []
        for src, tgt in batch:
            src_pos.append(torch.arange(1, len(src) + 1))
            tgt_pos.append(torch.arange(1, len(tgt) + 1))

            src_batch.append(torch.tensor(src))
            tgt_batch.append(torch.tensor(tgt))

        src_pos = pad_sequence(src_pos, batch_first=True, padding_value=0)
        tgt_pos = pad_sequence(tgt_pos, batch_first=True, padding_value=0)
        src_batch = pad_sequence(src_batch, batch_first=True, padding_value=PAD_idx)
        tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=PAD_idx)
        return {"src": src_batch, "src_pos": src_pos, "tgt": tgt_batch, "tgt_pos": tgt_pos}

    return collate

import os
from argparse import ArgumentParser
from pathlib import Path

import torch
import torch.nn as nn
import wandb
from sacrebleu.metrics import BLEU
from tokenizers import Tokenizer
from torch import optim
from tqdm import trange, tqdm
from torch.utils.data import DataLoader

from config import VASWANI_CONFIG, MASK_PREDICT_CONFIG
from src.models.decoding import translate
from src.models.mask_predict.loss import MaskPredictLoss
from src.models.mask_predict.model import MaskPredictModel
from src.models.vaswani.model import TranslationModel
from src.data.data import TranslationDataset, SpecialTokens, mask_predict_collator
from src.models.vaswani.loss import VaswaniLoss
from utils import DEVICE, create_mask, Dict, fix_seed


def train_epoch(
        model: TranslationModel,
        train_dataloader,
        criterion,
        optimizer,
        scheduler,
        cfg: Dict,
        progress_bar,
        PAD_IDX
):
    # train the model for one epoch
    # you can obviously add new arguments or change the API if it does not suit you
    model.train()
    losses = 0
    prefix = "epoch_"
    for step, batch in enumerate(train_dataloader):
        device_batch = {}
        for key in batch.keys():
            device_batch[key] = batch[key].to(device=DEVICE, dtype=torch.long, non_blocking=True, copy=False)

        tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(**device_batch, device=DEVICE, PAD_IDX=PAD_IDX)

        model_output = model(**device_batch, tgt_mask=tgt_mask,
                       src_padding_mask=src_padding_mask, tgt_padding_mask=tgt_padding_mask)
        loss_dict = criterion(model_output, device_batch)
        loss = loss_dict["loss"]
        # print(loss.item(), loss_dict["len_loss"], loss_dict["translate_loss"])
        try:
            loss.backward()
        except:
            print("AAA")

        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_thresh)

        if step % cfg.collect_batch == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            progress_bar.update(1)
            losses += loss.item()
            scheduler.step()

        if step % cfg.log_step == 0:
            wandb_log = {}
            for key in loss_dict:
                wandb_log[prefix + key] = loss_dict[key].item()
            # wandb.log(wandb_log)

    return losses / (len(train_dataloader) // cfg.collect_batch)


@torch.inference_mode()
def evaluate(model: TranslationModel, val_dataloader, criterion, cfg, PAD_IDX):
    # compute the loss over the entire validation subset
    model.eval()
    losses = None
    prefix = "val_"
    for step, batch in enumerate(val_dataloader):
        device_batch = {}
        for key in batch.keys():
            device_batch[key] = batch[key].to(device=DEVICE, dtype=torch.long, non_blocking=True, copy=False)

        tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(**device_batch, device=DEVICE, PAD_IDX=PAD_IDX)

        model_output = model(**device_batch, tgt_mask=tgt_mask,
                             src_padding_mask=src_padding_mask, tgt_padding_mask=tgt_padding_mask)
        loss_dict = criterion(model_output, device_batch)
        if losses is None:
            losses = {prefix + k: 0 for k in loss_dict.keys()}
        for key in loss_dict.keys():
            losses[prefix + key] += loss_dict[key].item()
    for key in losses.keys():
        losses[key] /= len(val_dataloader)
    return losses


def translate_test_set(model: TranslationModel, data_dir, output_dir, tokenizer_path):
    model.eval()
    src_tokenizer = Tokenizer.from_file(os.path.join(tokenizer_path, "tokenizer_de.json"))
    tgt_tokenizer = Tokenizer.from_file(os.path.join(tokenizer_path, "tokenizer_en.json"))

    greedy_translations = []
    with open(data_dir / "test.de.txt") as input_file, open(
            output_dir / "answers_greedy.txt", "w+"
    ) as output_file:
        test_sentences = input_file.readlines()
        greedy_translations = translate(model, test_sentences, src_tokenizer, tgt_tokenizer, "greedy", DEVICE)
        output_file.writelines(greedy_translations)
        output_file.flush()

    beam_translations = []
    with open(data_dir / "test.de.txt") as input_file, open(
            "answers_beam.txt", "w+"
    ) as output_file:
        # translate with beam search
        pass

    with open(data_dir / "test.en.txt") as input_file:
        references = [line.strip() for line in input_file]

    bleu = BLEU()
    bleu_greedy = bleu.corpus_score(greedy_translations, [references]).score

    # we're recreating the object, as it might cache some stats
    bleu = BLEU()
    bleu_beam = bleu.corpus_score(beam_translations, [references]).score

    print(f"BLEU with greedy search: {bleu_greedy}, with beam search: {bleu_beam}")
    # maybe log to wandb/comet/neptune as well


def define_model(model_type, model_cfg, data_cfg, **kwargs):
    if model_type == "vaswani":
        model = TranslationModel(
            num_encoder_layers=model_cfg.base_settings.num_encoder_layers,
            num_decoder_layers=model_cfg.base_settings.num_decoder_layers,
            emb_size=model_cfg.base_settings.emb_size,
            dim_feedforward=model_cfg.base_settings.dim_feedforward,
            n_head=model_cfg.base_settings.attention_heads,
            src_vocab_size=model_cfg.base_settings.vocab_size,
            tgt_vocab_size=model_cfg.base_settings.vocab_size,
            dropout_prob=model_cfg.base_settings.dropout_prob,
            n_positions=data_cfg.max_len
        )
        PAD_IDX = kwargs.get("loss_ignore", 1)
        loss_fn = VaswaniLoss(ignore_id=PAD_IDX)
        collator = {
            "train_collator": kwargs.get("train_collator"),
            "val_collator": kwargs.get("val_collator"),
        }
    elif model_type == "mask_predict":
        model = MaskPredictModel(vocab_size=model_cfg.base_settings.vocab_size,
                                 src_max_len=data_cfg.max_len, tgt_max_len=data_cfg.max_len,
                                 n_layers_enc=model_cfg.base_settings.num_encoder_layers,
                                 n_layers_dec=model_cfg.base_settings.num_decoder_layers,
                                 d_model=model_cfg.base_settings.emb_size,
                                 d_inner=model_cfg.base_settings.dim_feedforward,
                                 n_heads=model_cfg.base_settings.attention_heads,
                                 device=DEVICE, dropout=model_cfg.base_settings.dropout_prob)
        PAD_IDX = kwargs.get("loss_ignore", 1)
        train_data_path = kwargs.get("train_data_path")
        loss_fn = MaskPredictLoss(ignore_id=PAD_IDX, train_data_path=train_data_path, device=DEVICE)
        collate_fn = kwargs.get("mask_predict_collator")
        collate_fn = collate_fn(kwargs.get("tokenizer"))
        collator = {
            "train_collator": collate_fn,
            "val_collator": collate_fn,
        }

    return model, loss_fn, collator


def train_model(data_dir, tokenizer_path, num_epochs, model_type, cfg):
    src_tokenizer = Tokenizer.from_file(os.path.join(tokenizer_path, "tokenizer_de.json"))
    tgt_tokenizer = Tokenizer.from_file(os.path.join(tokenizer_path, "tokenizer_en.json"))
    PAD_IDX = src_tokenizer.token_to_id(SpecialTokens.PADDING.value)

    model_cfg = cfg.model
    data_cfg = cfg.data
    opt_cfg = model_cfg.optimizer
    scheduler_cfg = model_cfg.scheduler

    train_dataset = TranslationDataset(
        data_dir / "train.de.txt",
        data_dir / "train.en.txt",
        src_tokenizer,
        tgt_tokenizer,
        max_len=data_cfg.max_len,  # might be enough at first
    )
    val_dataset = TranslationDataset(
        data_dir / "val.de.txt",
        data_dir / "val.en.txt",
        src_tokenizer,
        tgt_tokenizer,
        max_len=data_cfg.max_len,
    )

    device = DEVICE
    fix_seed(cfg.seed)
    total_steps = num_epochs * len(train_dataset) // (cfg.data.batch_size_train * cfg.train.collect_batch)

    # create loss, optimizer, scheduler objects, dataloaders etc.
    # don't forget about collate_fn
    # if you intend to use AMP, you might need something else
    model, loss_fn, collator = define_model(model_type, model_cfg, data_cfg, loss_ignore=PAD_IDX,
                                            train_collator=train_dataset.collate_translation_data,
                                            val_collator=val_dataset.collate_translation_data,
                                            mask_predict_collator=mask_predict_collator,
                                            tokenizer=src_tokenizer, train_data_path=data_dir / "train.en.txt")
    model.to(device)
    print("Number of parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    train_dataloader = DataLoader(train_dataset, batch_size=data_cfg.batch_size_train,
                                  shuffle=True, num_workers=data_cfg.dataloader_num_workers, pin_memory=True,
                                  collate_fn=collator["train_collator"])
    val_dataloader = DataLoader(val_dataset, batch_size=data_cfg.batch_size_val, shuffle=False,
                                num_workers=data_cfg.dataloader_num_workers, pin_memory=False,
                                collate_fn=collator["val_collator"])

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, **opt_cfg)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, **scheduler_cfg, total_steps=total_steps)

    min_val_loss = float("inf")
    last_epoch = 1

    progress = tqdm(total=total_steps)
    # wandb.watch(model, optimizer, log="all", log_freq=10)
    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(model, train_dataloader, loss_fn,
                                 optimizer, scheduler, cfg.train, progress, PAD_IDX)
        val_loss_dict = evaluate(model, val_dataloader, loss_fn, cfg.train, PAD_IDX)

        # might be useful to translate some sentences from validation to check your decoding implementation
        idx = torch.randint(0, len(val_dataset), (10,))
        val_sentences = [val_dataset.src_tokenizer.decode(val_dataset[i][0]) for i in idx]
        val_targets = [val_dataset.tgt_tokenizer.decode(val_dataset[i][1]) for i in idx]
        translation = translate(model, val_sentences,
                                val_dataset.src_tokenizer, val_dataset.tgt_tokenizer,
                                "mask_predict", DEVICE, data_cfg)
        bleu_greedy = BLEU().corpus_score(translation, [val_targets]).score
        wandb_log = {"train_loss": train_loss, "BLEU": bleu_greedy}
        wandb_log.update(val_loss_dict)
        # wandb.log(wandb_log)

        # also, save the best checkpoint somewhere around here
        if val_loss_dict["val_loss"] < min_val_loss:
            os.makedirs(cfg.misc.checkpoint_path, exist_ok=True)
            torch.save({"model": model.state_dict(),
                        "opt": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict()},
                       os.path.join(cfg.misc.checkpoint_path, f"checkpoint_{epoch}.pth"))
            min_val_loss = val_loss_dict["val_loss"]

        # and the last one in case you need to recover
        # by the way, is this sufficient?
        torch.save({"model": model.state_dict(),
                    "opt": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict()},
                   os.path.join(cfg.misc.checkpoint_path, "checkpoint_last.pth"))
        last_epoch = epoch

    # load the best checkpoint
    model.load_state_dict(torch.load(os.path.join(cfg.misc.checkpoint_path, f"checkpoint_{last_epoch}.pth")))
    return model


if __name__ == "__main__":
    parser = ArgumentParser()
    data_group = parser.add_argument_group("Data paths")
    data_group.add_argument(
        "--data-dir", type=Path, help="Path to the directory containing processed data"
    )
    data_group.add_argument(
        "--tokenizer-path", type=Path, help="Path to the trained tokenizer files"
    )

    # argument groups are useful for separating semantically different parameters
    hparams_group = parser.add_argument_group("Training hyperparameters")
    hparams_group.add_argument(
        "--num-epochs", type=int, default=93, help="Number of training epochs"
    )
    hparams_group.add_argument(
        "--model-type", type=str, default="vaswani", help="Model architecture to train"
    )

    args = parser.parse_args()
    if args.model_type == "vaswani":
        cfg = VASWANI_CONFIG
    elif args.model_type == "mask_predict":
        cfg = MASK_PREDICT_CONFIG

    # with wandb.init(project="machine_translation", entity="johan_ddc_team", config=cfg):
    model = train_model(args.data_dir, args.tokenizer_path, args.num_epochs,
                        args.model_type, cfg)
    translate_test_set(model, args.data_dir, "data/results", args.tokenizer_path)

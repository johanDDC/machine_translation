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

from config import VASWANI_CONFIG as cfg
from src.models.decoding import translate
from src.models.vaswani import TranslationModel
from src.data.data import TranslationDataset, SpecialTokens
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
    for step, batch in enumerate(train_dataloader):
        device_batch = {}
        for key in batch.keys():
            device_batch[key] = batch[key].to(device=DEVICE, dtype=torch.long, non_blocking=True, copy=False)

        tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(**device_batch, device=DEVICE, PAD_IDX=PAD_IDX)

        logits = model(**device_batch, tgt_mask=tgt_mask,
                       src_padding_mask=src_padding_mask, tgt_padding_mask=tgt_padding_mask)
        loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt[:, 1:].reshape(-1))
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_thresh)

        if step % cfg.collect_batch == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            progress_bar.update(1)
            losses += loss.item()
            scheduler.step()

        if step % cfg.log_step == 0:
            wandb_log = {
                "epoch_loss": loss.item(),
                "epoch_perplexity": torch.exp(-loss)
            }
            wandb.log(wandb_log)
            model.train()

    return losses / (len(train_dataloader) // cfg.collect_batch)


@torch.inference_mode()
def evaluate(model: TranslationModel, val_dataloader, criterion, cfg, PAD_IDX):
    # compute the loss over the entire validation subset
    model.eval()
    losses = 0
    for step, batch in enumerate(val_dataloader):
        src = batch["src"].to(device=DEVICE, dtype=torch.long, non_blocking=True, copy=False)
        tgt = batch["tgt"].to(device=DEVICE, dtype=torch.long, non_blocking=True, copy=False)
        src_pos = batch["src_pos"].to(device=DEVICE, dtype=torch.long, non_blocking=True, copy=False)
        tgt_pos = batch["tgt_pos"].to(device=DEVICE, dtype=torch.long, non_blocking=True, copy=False)

        tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt[:, :-1], DEVICE, PAD_IDX)

        logits = model(src, src_pos, tgt[:, :-1], tgt_pos[:, :-1], tgt_mask, src_padding_mask, tgt_padding_mask)
        loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt[:, 1:].reshape(-1))
        losses += loss.item()
    return losses / len(val_dataloader)


def train_model(data_dir, tokenizer_path, num_epochs):
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
    model.to(device)
    print("Number of parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # create loss, optimizer, scheduler objects, dataloaders etc.
    # don't forget about collate_fn
    # if you intend to use AMP, you might need something else
    train_dataloader = DataLoader(train_dataset, batch_size=data_cfg.batch_size_train,
                                  shuffle=True, num_workers=data_cfg.dataloader_num_workers, pin_memory=True,
                                  collate_fn=train_dataset.collate_translation_data)
    val_dataloader = DataLoader(val_dataset, batch_size=data_cfg.batch_size_val, shuffle=False,
                                num_workers=data_cfg.dataloader_num_workers, pin_memory=False,
                                collate_fn=val_dataset.collate_translation_data)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, **opt_cfg)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, **scheduler_cfg, total_steps=total_steps)

    min_val_loss = float("inf")
    last_epoch = 1

    progress = tqdm(total=total_steps)
    wandb.watch(model, optimizer, log="all", log_freq=10)
    text_table = wandb.Table(columns=["translation", "prediction"])
    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(model, train_dataloader, loss_fn,
                                 optimizer, scheduler, cfg.train, progress, PAD_IDX)
        val_loss = evaluate(model, val_dataloader, loss_fn, cfg.train, PAD_IDX)

        # might be useful to translate some sentences from validation to check your decoding implementation
        idx = torch.randint(0, len(val_dataset), 10)
        val_sentences = [val_dataset.src_tokenizer.decode(val_dataset[i][0]) for i in idx]
        val_targets = [val_dataset.src_tokenizer.decode(val_dataset[i][1]) for i in idx]
        translation = translate(model, val_sentences,
                                val_dataset.src_tokenizer, val_dataset.tgt_tokenizer,
                                "greedy", DEVICE)
        bleu_greedy = BLEU().corpus_score(translation, [val_targets]).score
        text_table.add_data(val_targets[0], translation[0])
        wandb_log = {"train_loss": train_loss, "val_loss": val_loss,
                     "predictions": text_table, "BLEU": bleu_greedy}
        wandb.log(wandb_log)

        # also, save the best checkpoint somewhere around here
        if val_loss < min_val_loss:
            os.makedirs(cfg.misc.checkpoint_path, exist_ok=True)
            torch.save({"model": model.state_dict(),
                        "opt": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict()},
                       os.path.join(cfg.misc.checkpoint_path, f"checkpoint_{epoch}.pth"))
            min_val_loss = val_loss

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
        "--num-epochs", type=int, default=25, help="Number of training epochs"
    )

    args = parser.parse_args()

    with wandb.init(project="machine_translation", entity="johan_ddc_team", config=cfg):
        model = train_model(args.data_dir, args.tokenizer_path, args.num_epochs)
    translate_test_set(model, args.data_dir, args.tokenizer_path)

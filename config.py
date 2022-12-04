from utils import Dict

_BASE_CONFIG = {
    "seed": 322,
    "train": {
        "grad_clip_thresh": 1,
        "collect_batch": 1,
        "log_step": 300
    },
    "data": {
        "batch_size_train": 64,
        "batch_size_val": 64,
        "dataloader_num_workers": 8,
        "max_len": 128
    },
    "misc": {
        "checkpoint_path": "./checkpoints"
    }
}

VASWANI_CONFIG = {
    **_BASE_CONFIG,
    "model": {
        "base_settings": {
            "vocab_size": 30_000,
            "num_encoder_layers": 6,
            "num_decoder_layers": 6,
            "emb_size": 256,
            "dim_feedforward": 512,
            "attention_heads": 4,
            "dropout_prob": 0.1
        },
        "optimizer": {
            "betas": (0.9, 0.98),
            "eps": 1e-9,
            "weight_decay": 1e-6,
        },
        "scheduler": {
            "max_lr": 3e-4,
            "div_factor": 10_000,
            "pct_start": 0.1,
            "anneal_strategy": "cos"
        }
    },
}

VASWANI_CONFIG = Dict(VASWANI_CONFIG)

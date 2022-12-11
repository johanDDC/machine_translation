from collections import defaultdict

import torch
from torch import nn
import torch.nn.functional as F


class MaskPredictLoss(nn.Module):
    def __init__(self, ignore_id, train_data_path, num_classes=128, device="cpu"):
        super().__init__()
        self.ignoe_idx = ignore_id
        self.weights = self.__compute_len_weights(train_data_path, num_classes)
        self.weights = self.weights.to(device)

    @staticmethod
    def __compute_len_weights(path, num_classes):
        weights = torch.zeros((num_classes + 1,), dtype=torch.float32)
        d = defaultdict(int)
        with open(path, "r") as f:
            lines = f.readlines()
            num_records = len(lines)
            for line in lines:
                d[len(line.split())] += 1
        for cls in d.keys():
            class_count = d[cls]
            weights[cls] = num_records / (num_classes * class_count)
        return weights


    def forward(self, model_output, batch):
        logits_translate = model_output["output"]
        logits_length = model_output["len_prediction"]

        target_unmasked = batch["tgt_unmasked"]
        target_len = batch["tgt_lens"]

        translate_probs = F.log_softmax(logits_translate, -1)

        len_loss = F.cross_entropy(logits_length, target_len, weight=self.weights)
        translate_loss = F.nll_loss(translate_probs.view(-1, translate_probs.shape[-1]),
                                    target_unmasked.view(-1),
                                    ignore_index=self.ignoe_idx)

        return {
            "len_loss": len_loss,
            "translate_loss": translate_loss,
            "loss": 1.1 * len_loss + translate_loss
        }

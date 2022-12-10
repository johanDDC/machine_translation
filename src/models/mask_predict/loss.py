import torch
from torch import nn
import torch.nn.functional as F


class MaskPredictLoss(nn.Module):
    def __init__(self, ignore_id, len_confidence=0.5, len_candidates=5, num_classes=128, device="cpu"):
        super().__init__()
        self.ignoe_idx = ignore_id
        self.len_confidence = len_confidence
        self.len_candidates = len_candidates
        self.weight = torch.ones((num_classes,))
        for i in range(1, 5):
            self.weight[[i, -i]] = 2 - (i - 1) / 4
        self.weight[0] = 0
        self.weight = self.weight.to(device)


    def forward(self, model_output, batch):
        logits_translate = model_output["output"]
        logits_length = model_output["len_prediction"]

        target_unmasked = batch["tgt_unmasked"]
        target_len = batch["tgt_lens"]

        translate_probs = F.log_softmax(logits_translate, -1)

        # len_probs = F.log_softmax(logits_length, -1)
        # len_distributions = torch.zeros_like(logits_length, device=logits_length.device)
        # reserve = (1 - self.len_confidence) / 2
        # one_side = (self.len_candidates - 1) // 2
        # x_ = reserve / sum([1.5 ** pow for pow in range(one_side)])
        #
        # central_l = target_len[(target_len - one_side > 0) & (target_len + one_side < len_distributions.shape[1])]
        # corner_left_l = target_len[(target_len - 1) <= 0]
        # pre_corner_left_l = target_len[(target_len - one_side <= 0) & (target_len - 1 > 0)]
        # corner_right_l = target_len[(target_len + 1 >= len_distributions.shape[1])]
        # pre_corner_right_l = target_len[(target_len + one_side >= len_distributions.shape[1]) & (target_len + 1 < len_distributions.shape[1])]
        #
        # len_distributions.scatter_(1, central_l.view(-1, 1), torch.ones_like(central_l).view(-1, 1) * 0.5)
        # len_distributions.scatter_(1, corner_left_l.view(-1, 1), torch.ones_like(corner_left_l).view(-1, 1) * (0.5 + 2.5 * x_))
        # len_distributions.scatter_(1, pre_corner_left_l.view(-1, 1), torch.ones_like(pre_corner_left_l).view(-1, 1) * (0.5 + x_))
        # len_distributions.scatter_(1, corner_right_l.view(-1, 1), torch.ones_like(corner_right_l).view(-1, 1) * (0.5 + 2.5 * x_))
        # len_distributions.scatter_(1, pre_corner_right_l.view(-1, 1), torch.ones_like(pre_corner_right_l).view(-1, 1) * (0.5 + x_))
        # for i in range(one_side - 1, -1, -1):
        #     len_distributions.scatter_(1, (central_l - (one_side - i)).view(-1, 1), torch.ones_like(central_l).view(-1, 1) * 1.5 ** i * x_)
        #     len_distributions.scatter_(1, (central_l + (one_side - i)).view(-1, 1), torch.ones_like(central_l).view(-1, 1) * 1.5 ** i * x_)
        #
        # len_distributions.scatter_(1, (pre_corner_left_l - 1).view(-1, 1), torch.ones_like(pre_corner_left_l).view(-1, 1) * 1.5 * x_)
        # len_distributions.scatter_(1, (pre_corner_left_l + 1).view(-1, 1), torch.ones_like(pre_corner_left_l).view(-1, 1) * 1.5 * x_)
        # len_distributions.scatter_(1, (pre_corner_left_l + 2).view(-1, 1), torch.ones_like(pre_corner_left_l).view(-1, 1) * x_)
        #
        # len_distributions.scatter_(1, (pre_corner_right_l - 1).view(-1, 1), torch.ones_like(pre_corner_right_l).view(-1, 1) * 1.5 * x_)
        # len_distributions.scatter_(1, (pre_corner_right_l - 2).view(-1, 1), torch.ones_like(pre_corner_right_l).view(-1, 1) * x_)
        # len_distributions.scatter_(1, (pre_corner_right_l + 1).view(-1, 1), torch.ones_like(pre_corner_right_l).view(-1, 1) * 1.5 * x_)
        #
        # len_distributions.scatter_(1, (corner_right_l - 1).view(-1, 1), torch.ones_like(corner_right_l).view(-1, 1) * 1.5 * x_)
        # len_distributions.scatter_(1, (corner_right_l - 2).view(-1, 1), torch.ones_like(corner_right_l).view(-1, 1) * x_)
        #
        # len_distributions.scatter_(1, (corner_left_l + 1).view(-1, 1), torch.ones_like(corner_left_l).view(-1, 1) * 1.5 * x_)
        # len_distributions.scatter_(1, (corner_left_l + 2).view(-1, 1), torch.ones_like(corner_left_l).view(-1, 1) * x_)

        len_loss = F.cross_entropy(logits_length, target_len, label_smoothing=0.3, weight=self.weight)
        translate_loss = F.nll_loss(translate_probs.view(-1, translate_probs.shape[-1]),
                                    target_unmasked.view(-1),
                                    ignore_index=self.ignoe_idx)

        return {
            "len_loss": len_loss,
            "translate_loss": translate_loss,
            "loss": 1.1 * len_loss + translate_loss
        }

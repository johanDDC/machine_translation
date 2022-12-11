import torch
from torch import nn

class VaswaniLoss(nn.Module):
    def __init__(self, ignore_id):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_id)

    def forward(self, model_output, batch):
        target = batch["tgt_seq"]
        loss = self.loss_fn(model_output.reshape(-1, model_output.shape[-1]),
                            target[:, 1:].reshape(-1))
        return {"loss": loss}


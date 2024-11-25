import torch
import torch.nn.functional as F

from model.method.Method import Method

class Classic(Method):
    def __init__(self) -> None:
        super().__init__()

    def train_batch(self, architecture, x, y):
        pred = architecture(x)

        loss = F.mse_loss(pred, y)

        return loss

    def inference(self, architectur, x) -> torch.Tensor:
        return architectur(x)

import torch
import torchmetrics
from torch import Tensor
from torchmetrics.classification import BinaryAUROC


class PartialAUC(torchmetrics.Metric):
    def __init__(self, min_tpr: float = 0.8):
        super(PartialAUC, self).__init__()
        self._min_tpr = min_tpr
        self.reset()

    def update(self, preds: Tensor, target: Tensor) -> None:
        self._gts = torch.cat([self._gts, target], 0)
        self._preds = torch.cat([self._preds, preds], 0)

    def compute(self) -> torch.Tensor:
        v_gt = torch.abs(self._gts - 1)
        v_pred = 1.0 - self._preds
        max_fpr = abs(1 - self._min_tpr)
        partial_auc_scaled = BinaryAUROC(max_fpr=max_fpr)(v_pred, v_gt)
        partial_auc = 0.5 * max_fpr**2 + (max_fpr - 0.5 * max_fpr**2) / (1.0 - 0.5) * (
            partial_auc_scaled - 0.5
        )
        return partial_auc

    def reset(self) -> None:
        self._gts = torch.empty(0, dtype=torch.float32)
        self._preds = torch.empty(0, dtype=torch.float32)

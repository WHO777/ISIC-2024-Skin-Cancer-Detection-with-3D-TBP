import inspect
import sys
from typing import Dict, List, Sequence, Union

import lightning
import lightning.pytorch as pl
import torch
import torchmetrics
from lightning import Callback
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from omegaconf import DictConfig

from src.isic import metrics, models


class ISIC2024LightningModule(lightning.LightningModule):
    def __init__(
        self,
        model_cfg: DictConfig,
        loss_cfg: DictConfig,
        optimizer_cfg: DictConfig,
        callbacks_cfg: List[DictConfig],
        metrics_cfg: List[DictConfig],
    ):
        super(ISIC2024LightningModule, self).__init__()
        self._model_cfg = model_cfg
        self._loss_cfg = loss_cfg
        self._optimizer_cfg = optimizer_cfg
        self._callbacks_cfg = callbacks_cfg
        self._metrics_cfg = metrics_cfg

        self._compile_model()

        self._training_step_outputs = []
        self._validation_step_outputs = []

    def _compile_model(self) -> None:
        model_type = self._model_cfg["type"]
        self._model = getattr(models, model_type)(**self._model_cfg["params"])
        self._loss = self._build_loss()
        self._metrics = self._build_metrics()

    def _build_loss(self) -> torch.nn.Module:
        torch_losses = {
            name: obj
            for name, obj in inspect.getmembers(sys.modules[torch.nn.__name__])
            if inspect.isclass(obj) and "Loss" in name
        }
        torchmetrics_losses = {
            name: obj
            for name, obj in inspect.getmembers(
                sys.modules[torchmetrics.regression.__name__]
            )
            if inspect.isclass(obj)
        }
        available_loss_types = {**torch_losses, **torchmetrics_losses}
        loss_type = self._loss_cfg["type"]
        loss_params = self._loss_cfg.get("params", {})
        loss = available_loss_types[loss_type](**loss_params)
        return loss

    def _build_metrics(self) -> Dict[str, torchmetrics.Metric]:
        metrics_ = {}
        for metric_cfg in self._metrics_cfg:
            metric_type = metric_cfg["type"]
            metric_params = metric_cfg.get("params", {})
            try:
                metric = getattr(torchmetrics.classification, metric_type)(
                    **metric_params
                )
            except AttributeError:
                metric = getattr(metrics, metric_type)(**metric_params)
            metric_name = (
                metric_cfg.name if "name" in metric_cfg else metric.__class__.__name__
            )
            metrics_[metric_name] = metric.to(self.device)
        return metrics_

    def forward(self, inputs):
        outputs = self._model(inputs["image"]).squeeze(1)
        return outputs

    def training_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        loss = self._loss(outputs, batch["target"])
        self._training_step_outputs.append(
            {"target": batch["target"].cpu(), "outputs": outputs.detach().cpu()}
        )
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self) -> None:
        for tso in self._training_step_outputs:
            for metric_name in self._metrics.keys():
                self._metrics[metric_name].update(tso["outputs"], tso["target"])
        metric_scores = {
            "train_" + name: metric.compute().item()
            for name, metric in self._metrics.items()
        }
        self.log_dict(metric_scores, on_epoch=True, prog_bar=True)
        self._training_step_outputs.clear()
        self._reset_metric_states()

    def validation_step(self, batch, batch_idx) -> None:
        outputs = self.forward(batch)
        loss = self._loss(outputs, batch["target"])
        self._validation_step_outputs.append(
            {"target": batch["target"].cpu(), "outputs": outputs.detach().cpu()}
        )
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        if not len(self._validation_step_outputs):
            return
        for vso in self._validation_step_outputs:
            for metric_name in self._metrics.keys():
                self._metrics[metric_name].update(vso["outputs"], vso["target"])
        metric_scores = {
            "val_" + name: metric.compute().item()
            for name, metric in self._metrics.items()
        }
        self.log_dict(metric_scores, on_epoch=True, prog_bar=True, sync_dist=True)
        self._validation_step_outputs.clear()
        self._reset_metric_states()

    def _reset_metric_states(self) -> None:
        for metric_name in self._metrics.keys():
            self._metrics[metric_name].reset()

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer_type = self._optimizer_cfg["type"]
        optimizer_params = self._optimizer_cfg.get("params")
        optimizer = getattr(torch.optim, optimizer_type)(
            self.parameters(), **optimizer_params
        )

        scheduler_cfg = self._optimizer_cfg.get("scheduler")
        if scheduler_cfg is not None:
            scheduler_type = scheduler_cfg["type"]
            scheduler_params = scheduler_cfg.get("params", {})
            scheduler = getattr(torch.optim.lr_scheduler, scheduler_type)(
                optimizer, **scheduler_params
            )
        else:
            scheduler = None
        return [optimizer], [scheduler]

    def configure_callbacks(self) -> Union[Sequence[Callback], Callback]:
        callbacks_ = []
        for callback_cfg in self._callbacks_cfg:
            callback_type = callback_cfg["type"]
            callback_params = callback_cfg.get("params", {})
            callback = getattr(pl.callbacks, callback_type)(**callback_params)
            callbacks_.append(callback)
        return callbacks_

import copy
from typing import Optional

import albumentations as A
from albumentations.pytorch import ToTensorV2
from lightning import LightningDataModule
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.isic.datasets.clearml import ClearMLDataset
from src.isic.datasets.isic2024 import ConcatISIC2024Dataset, ISIC2024Dataset

DATASET_TYPE_TO_CLASS_MAP = {
    "ClearMLDataset": ClearMLDataset,
    "ISIC2024Dataset": ISIC2024Dataset,
}


class ISIC2024DataModule(LightningDataModule):
    def __init__(self, data_cfg: DictConfig):
        super(ISIC2024DataModule, self).__init__()
        self._data_cfg = copy.deepcopy(data_cfg)

        train_transforms = eval(self._data_cfg["train_transforms"], {"A": A}) + [
            ToTensorV2()
        ]
        train_transforms = A.Compose(train_transforms)

        val_transforms = None
        if "val_transforms" in self._data_cfg:
            val_transforms = eval(self._data_cfg["val_transforms"], {"A": A}) + [
                ToTensorV2()
            ]
            val_transforms = A.Compose(val_transforms)

        ds_cfgs = [cfg for cfg in self._data_cfg.dataset.values()]
        ds_classes = [DATASET_TYPE_TO_CLASS_MAP[ds_cfg.type] for ds_cfg in ds_cfgs]

        train_datasets = [
            ds_class(**ds_cfg.params, transforms=train_transforms, train=True)
            for ds_class, ds_cfg in zip(ds_classes, ds_cfgs)
        ]
        self._train_ds = ConcatISIC2024Dataset(train_datasets)

        self._val_ds = None
        if self._data_cfg.get("val_dataloader") is not None:
            val_datasets = [
                ds_class(**ds_cfg.params, transforms=val_transforms, train=False)
                for ds_class, ds_cfg in zip(ds_classes, ds_cfgs)
            ]
            self._val_ds = ConcatISIC2024Dataset(val_datasets)

    def train_dataloader(self) -> DataLoader:
        train_dataloader_cfg = self._data_cfg.train_dataloader
        train_dataloader = DataLoader(self._train_ds, **train_dataloader_cfg)
        return train_dataloader

    def val_dataloader(self) -> Optional[DataLoader]:
        if self._val_ds is not None:
            val_dataloader_cfg = self._data_cfg.val_dataloader
            val_dataloader = DataLoader(self._val_ds, **val_dataloader_cfg)
            return val_dataloader
        return None

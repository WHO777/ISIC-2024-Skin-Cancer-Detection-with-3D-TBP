from io import BytesIO
from pathlib import Path
from typing import Optional, Union

import albumentations as A
import h5py
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset


class ISIC2024Dataset(Dataset):
    def __init__(
        self,
        root_path: Union[Path, str],
        pos_neg_ratio: float = 1.0,
        train: bool = True,
        n_folds: Optional[int] = None,
        fold: Optional[int] = None,
        seed: int = 43,
        **kwargs,
    ):
        root_path = Path(root_path)
        df_path = root_path / "train-metadata.csv"

        df = pd.read_csv(df_path)

        images_filepath = root_path / "train-image.hdf5"
        file_hdf = h5py.File(images_filepath)
        df = df[df["isic_id"].isin(file_hdf.keys())]

        df_positive = df[df["target"] == 1]
        df_negative = df[df["target"] == 0]

        num_neg_samples = int(len(df_positive) / pos_neg_ratio)
        df_negative = df_negative.sample(frac=1, random_state=seed).iloc[
            :num_neg_samples
        ]
        df = (
            pd.concat([df_positive, df_negative])
            .sample(frac=1, random_state=seed)
            .reset_index(drop=True)
        )

        if not train and n_folds is None:
            raise ValueError(
                'To get the validation dataset you must specify "n_folds" and "fold".'
            )

        if n_folds is not None:
            skf = StratifiedKFold(n_splits=n_folds or 5)
            for fold, (train_idx, val_idx) in enumerate(skf.split(X=df, y=df.target)):
                df.loc[val_idx, "fold"] = fold
            if train:
                df = df[df["fold"] != fold]
            else:
                df = df[df["fold"] == fold]

        self._ds = _ISIC2024Dataset(df, file_hdf, **kwargs)

    def __len__(self):
        return self._ds.__len__()

    def __getitem__(self, idx):
        return self._ds.__getitem__(idx)


class ConcatISIC2024Dataset(Dataset):
    def __init__(
        self,
        datasets,
    ):
        super(ConcatISIC2024Dataset, self).__init__()
        self._ds = torch.utils.data.ConcatDataset(datasets)

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx):
        return self._ds.__getitem__(idx)


class _ISIC2024Dataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        file_hdf: h5py.File,
        transforms: Optional[A.Compose] = None,
    ):
        super(_ISIC2024Dataset, self).__init__()
        self._df = df
        self._file_hdf = file_hdf
        self._isic_ids = df["isic_id"].values
        self._targets = df["target"].values

        self._transforms = transforms

    def __len__(self):
        return len(self._df)

    def __getitem__(self, idx):
        isic_id = self._isic_ids[idx]
        image = np.array(Image.open(BytesIO(self._file_hdf[isic_id][()])))
        image = self._transforms(image=image)["image"]
        target = torch.tensor(self._targets[idx], dtype=torch.float32)
        return {"image": image, "target": target}

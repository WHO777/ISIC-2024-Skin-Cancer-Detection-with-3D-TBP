from clearml import Dataset as CMLDataset

from src.isic.datasets.isic2024 import ISIC2024Dataset


class ClearMLDataset(ISIC2024Dataset):
    def __init__(
        self, name: str, project: str, version: str = None, alias: str = None, **kwargs
    ):
        alias = alias or name
        ds_root_path = CMLDataset.get(
            dataset_name=name,
            dataset_project=project,
            dataset_version=version,
            alias=alias,
        ).get_local_copy()
        super(ClearMLDataset, self).__init__(ds_root_path, **kwargs)

import string
from pathlib import Path
from typing import List

import hydra
import lightning as L
from lightning.pytorch import loggers
from omegaconf import DictConfig

try:
    from clearml import Task

    clear_ml_available = True
except ImportError:
    clear_ml_available = False

from src.isic.datasets.data_module import ISIC2024DataModule
from src.isic.models.lightning_module import ISIC2024LightningModule

OUTPUT_ROOT = Path("runs") / "train"
OUTPUT_ROOT.mkdir(exist_ok=True, parents=True)
DEFAULT_RUN_NAME = "exp0"


def generate_unique_run_name(output_dir, run_name: str) -> str:
    run_name = run_name or DEFAULT_RUN_NAME
    run_name = run_name.rstrip(string.digits)
    existing_run_names = [p.name for p in list(output_dir.iterdir())]
    if run_name not in existing_run_names:
        return run_name
    run_number = 1
    while (output_dir / (run_name + str(run_number))).is_dir():
        run_number += 1
    run_name = f"{run_name}{run_number}"
    return run_name


def get_loggers(loggers_cfg: List[DictConfig]):
    loggers_ = []
    for logger_cfg in loggers_cfg:
        logger_type = logger_cfg.type
        logger_params = logger_cfg.params
        logger = getattr(loggers, logger_type)(**logger_params)
        loggers_.append(logger)
    return loggers_


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    L.seed_everything(cfg.seed)

    output_dir = OUTPUT_ROOT / generate_unique_run_name(OUTPUT_ROOT, cfg.run_name)
    output_dir.mkdir(exist_ok=True, parents=True)
    cfg.output_dir = output_dir
    cfg.run_name = output_dir.name

    if cfg.clearML.enable:
        assert (
            clear_ml_available
        ), 'ClearML is not installed. You can install it using "pip3 install clearml"'
        task = Task.init(
            project_name=cfg.clearML.project_name,
            task_name=cfg.clearML.task_name,
            task_type=Task.TaskTypes.training,
        )

    data_module = ISIC2024DataModule(cfg.data)

    train_dataloader = data_module.train_dataloader()
    val_dataloader = data_module.val_dataloader()

    model = ISIC2024LightningModule(
        cfg.model, cfg.loss, cfg.optimizer, cfg.callbacks, cfg.metrics
    )

    loggers_cfg = cfg.get("loggers", [])
    loggers_ = get_loggers(loggers_cfg)

    trainer = L.Trainer(logger=loggers_)
    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == "__main__":
    main()

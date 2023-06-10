from enum import Enum, auto
from logging import INFO, getLogger
from pathlib import Path
from typing import Optional
import wandb

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

from label_selection_layer import do_experiment
from label_selection_layer.data_modules import PreprocessedNERMTurkDataModule
from label_selection_layer.models.ner_mturk_expt import BasicModel, CrowdLayerModel, LabelSelectionModel

logger = getLogger(__name__)
logger.setLevel(INFO)

torch.set_float32_matmul_precision("high")


class ModelType(Enum):
    BASIC = auto()
    CROWD_LAYER = auto()
    LABEL_SELECTION = auto()  # NOTE: This is the proposed methods.


class Executor:
    WANDB_PROJECT: str = "NerMturkExpr"
    DATASET_NAME: str = "ner_mturk"
    N_WORKERS: int = 47
    N_LABELS: int = 10
    NUM_EMBEDDINGS: int = 18_179

    @classmethod
    def _train_and_save(
        cls,
        experiment_name,
        model,
        data_module,
        max_epochs,
        savedir,
        early_stopping: bool,
        logger_type: Optional[str] = "wandb",
    ) -> bool:
        logger.info(f"[START] {cls.DATASET_NAME}-{experiment_name}")
        preds = do_experiment(
            data_module,
            model,
            experiment_name,
            max_epochs,
            logger_type=logger_type,
            wandb_project=cls.WANDB_PROJECT,
            should_stop_early=early_stopping,
        )
        preds = np.concatenate(preds)

        savedir.joinpath(cls.DATASET_NAME)
        savedir.mkdir(exist_ok=True)
        save_path = savedir.joinpath(f"{experiment_name}.npy")
        with save_path.open(mode="wb") as f:
            np.save(f, preds)
            logger.info(f'[SAVE] at "{save_path}"')

        logger.info(f"[DONE] {experiment_name}")
        return True

    @staticmethod
    def load_embeddings(datadir: Path):
        embeddings = np.load(datadir.joinpath("ner_mturk_embeddings.npy"))
        embeddings = torch.from_numpy(embeddings.astype(np.float32)).clone()
        return embeddings

    @classmethod
    def basic_model(
        cls,
        batch_size: int,
        max_epochs: int,
        label_type: str,
        savedir: Path,
        datadir: Path,
        early_stopping: bool,
        seed: Optional[int],
    ) -> bool:
        data_module = PreprocessedNERMTurkDataModule(
            datadir=datadir,
            label=label_type,
            batch_size=batch_size,
            num_workers=4,
        )
        model = BasicModel(
            num_embeddings=cls.NUM_EMBEDDINGS,
            embeddings=cls.load_embeddings(datadir),
            n_labels=cls.N_LABELS,
        )
        experiment_name = f"{type(model).__name__}_seed_{seed}_label_type_{label_type}"
        was_finished: bool = cls._train_and_save(
            experiment_name,
            model,
            data_module,
            max_epochs,
            savedir,
            early_stopping,
        )
        return was_finished

    @classmethod
    def crowd_layer_model(
        cls,
        batch_size: int,
        max_epochs: int,
        mode: str,
        savedir: Path,
        datadir: Path,
        early_stopping: bool,
        should_pretrain: bool = False,
        seed: Optional[int] = None,
    ) -> bool:
        model = CrowdLayerModel(
            num_embeddings=cls.NUM_EMBEDDINGS,
            embeddings=cls.load_embeddings(datadir),
            n_labels=cls.N_LABELS,
            n_workers=cls.N_WORKERS,
            crowd_layer_mode=mode,
        )

        experiment_name = f"{type(model).__name__}_{mode}_seed_{seed}_annotation"

        if should_pretrain:
            data_module_for_pretraining = PreprocessedNERMTurkDataModule(
                datadir=datadir,
                label="mv",
                batch_size=batch_size,
                num_workers=4,
            )

            model_for_pretraining = BasicModel(
                num_embeddings=cls.NUM_EMBEDDINGS,
                embeddings=cls.load_embeddings(datadir),
                n_labels=cls.N_LABELS,
            )

            _ = cls._train_and_save(
                experiment_name,
                model_for_pretraining,
                data_module_for_pretraining,
                5,
                savedir,
                early_stopping=False,
                logger_type="none",
            )

            model.load_state_dict(model_for_pretraining.state_dict(), strict=False)

        data_module = PreprocessedNERMTurkDataModule(datadir=datadir, label="annotation", batch_size=batch_size)
        was_finished: bool = cls._train_and_save(
            experiment_name,
            model,
            data_module,
            max_epochs,
            savedir,
            early_stopping,
        )
        return was_finished

    @classmethod
    def label_selection_model(
        cls,
        batch_size: int,
        max_epochs: int,
        mode: str,
        c: float,
        lambda_: int,
        savedir: Path,
        datadir: Path,
        early_stopping: bool,
        should_pretrain: bool,
        seed: Optional[int],
    ) -> bool:
        model = LabelSelectionModel(
            num_embeddings=cls.NUM_EMBEDDINGS,
            embeddings=cls.load_embeddings(datadir),
            n_labels=cls.N_LABELS,
            n_workers=cls.N_WORKERS,
            selective_mode=mode,
            c=c,
            lambda_=lambda_,
        )

        experiment_name = f"{type(model).__name__}_{mode}_seed_{seed}_annotation_c_{int(c*100):03}_"

        if should_pretrain:
            data_module_for_pretraining = PreprocessedNERMTurkDataModule(
                datadir=datadir,
                label="mv",
                batch_size=batch_size,
                num_workers=4,
            )

            model_for_pretraining = BasicModel(
                num_embeddings=cls.NUM_EMBEDDINGS,
                embeddings=cls.load_embeddings(datadir),
                n_labels=cls.N_LABELS,
            )

            _ = cls._train_and_save(
                experiment_name,
                model_for_pretraining,
                data_module_for_pretraining,
                5,
                savedir,
                early_stopping=False,
                logger_type="none",
            )

            model.load_state_dict(model_for_pretraining.state_dict(), strict=False)

        data_module = PreprocessedNERMTurkDataModule(datadir=datadir, label="annotation", batch_size=batch_size)
        was_finished: bool = cls._train_and_save(
            experiment_name,
            model,
            data_module,
            max_epochs,
            savedir,
            early_stopping,
        )
        return was_finished


@hydra.main(version_base="1.2", config_path=".", config_name="ner_mturk_config")
def main(cfg: DictConfig):
    model_type: ModelType = ModelType[cfg.model_type]
    datadir = Path("/home/ykosuke/github/experimental-tools/data/ner-mturk/preprocessed")
    savedir = Path(cfg.savedir).resolve()

    if cfg.seed is not None:
        pl.seed_everything(cfg.seed)

    if model_type == ModelType.BASIC:
        cfg_ = cfg.basic_model
        Executor.basic_model(
            cfg.batch_size,
            cfg.max_epochs,
            cfg_.label_type,
            savedir,
            datadir,
            early_stopping=cfg.early_stopping,
            seed=cfg.seed,
        )

    elif model_type == ModelType.CROWD_LAYER:
        cfg_ = cfg.crowd_layer_model
        Executor.crowd_layer_model(
            cfg.batch_size,
            cfg.max_epochs,
            cfg_.mode,
            savedir,
            datadir,
            should_pretrain=True,
            early_stopping=cfg.early_stopping,
            seed=cfg.seed,
        )

    elif model_type == ModelType.LABEL_SELECTION:
        cfg_ = cfg.label_selection_model
        Executor.label_selection_model(
            cfg.batch_size,
            cfg.max_epochs,
            cfg_.mode,
            cfg_.c,
            cfg_.lmd,
            savedir,
            datadir,
            should_pretrain=cfg_.should_pretrain,
            early_stopping=cfg.early_stopping,
            seed=cfg.seed,
        )

    else:
        logger.info("Did not execute an experiment or more.")

    wandb.finish()


if __name__ == "__main__":
    main()

from enum import Enum, auto
from logging import INFO, getLogger
from pathlib import Path
from typing import Optional

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

from label_selection_layer import do_experiment
from label_selection_layer.data_modules import PreprocessedMovieReviewsDataModule
from label_selection_layer.models.movie_reviews_expt import BasicModel, CrowdLayerModel, LabelSelectionModel

logger = getLogger(__name__)
logger.setLevel(INFO)


class ModelType(Enum):
    BASIC = auto()
    CROWD_LAYER = auto()
    LABEL_SELECTION = auto()  # NOTE: This is the proposed methods.


class Executor:
    WANDB_PROJECT: str = "MovieReviewExpr"
    DATASET_NAME: str = "movie_review"
    N_WORKERS: int = 135

    @classmethod
    def _train_and_save(cls, experiment_name, model, data_module, max_epochs, savedir) -> bool:
        logger.info(f"[START] {cls.DATASET_NAME}-{experiment_name}")
        preds = do_experiment(data_module, model, experiment_name, max_epochs, wandb_project=cls.WANDB_PROJECT)
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
        embeddings = np.load(datadir.joinpath("embedding_matrix.npy"))
        embeddings = torch.from_numpy(embeddings.astype(np.float32)).clone()
        return embeddings

    @classmethod
    def basic_model(
        cls,
        batch_size: int,
        max_epochs: int,
        num_embeddings: int,
        label_type: str,
        savedir: Path,
        datadir: Path,
        seed: Optional[int] = None,
    ) -> bool:
        if seed is not None:
            pl.seed_everything(seed)

        data_module = PreprocessedMovieReviewsDataModule(datadir=datadir, target=label_type, batch_size=batch_size)
        embeddings = cls.load_embeddings(datadir)
        model = BasicModel(num_embeddings=num_embeddings, embeddings=embeddings)
        experiment_name = f"{type(model).__name__}_seed_{seed}_label_type_{label_type}"
        return cls._train_and_save(experiment_name, model, data_module, max_epochs, savedir)

    @classmethod
    def crowd_layer_model(
        cls,
        batch_size: int,
        max_epochs: int,
        num_embeddings: int,
        mode: str,
        savedir: Path,
        datadir: Path,
        seed: Optional[int] = None,
    ) -> bool:
        if seed is not None:
            pl.seed_everything(seed)

        data_module = PreprocessedMovieReviewsDataModule(datadir=datadir, target="annotation", batch_size=batch_size)
        embeddings = cls.load_embeddings(datadir)
        model = CrowdLayerModel(
            num_embeddings=num_embeddings, embeddings=embeddings, n_workers=cls.N_WORKERS, crowd_layer_mode=mode
        )
        experiment_name = f"{type(model).__name__}_{mode}_seed_{seed}_annotation"
        return cls._train_and_save(experiment_name, model, data_module, max_epochs, savedir)

    @classmethod
    def label_selection_model(
        cls,
        batch_size: int,
        max_epochs: int,
        num_embeddings: int,
        mode: str,
        c: float,
        savedir: Path,
        datadir: Path,
        seed: Optional[int] = None,
    ) -> bool:
        if seed is not None:
            pl.seed_everything(seed)

        data_module = PreprocessedMovieReviewsDataModule(datadir=datadir, target="annotation", batch_size=batch_size)
        embeddings = cls.load_embeddings(datadir)
        model = LabelSelectionModel(
            num_embeddings=num_embeddings, embeddings=embeddings, n_workers=cls.N_WORKERS, selective_mode=mode, c=c
        )
        experiment_name = f"{type(model).__name__}_{mode}_seed_{seed}_annotation_c_{int(c*100):03}"
        return cls._train_and_save(experiment_name, model, data_module, max_epochs, savedir)


@hydra.main(version_base="1.2", config_path=".", config_name="movie_review_config")
def main(cfg: DictConfig):
    model_type: ModelType = ModelType[cfg.model_type]
    datadir = Path(cfg.datadir)
    savedir = Path(cfg.savedir).resolve()

    if model_type == ModelType.BASIC:
        cfg_ = cfg.basic_model
        Executor.basic_model(
            cfg.batch_size,
            cfg.max_epochs,
            cfg.num_embeddings,
            cfg_.label_type,
            savedir,
            datadir,
            seed=cfg.seed,
        )

    elif model_type == ModelType.CROWD_LAYER:
        cfg_ = cfg.crowd_layer_model
        Executor.crowd_layer_model(
            cfg.batch_size,
            cfg.max_epochs,
            cfg.num_embeddings,
            cfg_.mode,
            savedir,
            datadir,
            seed=cfg.seed,
        )

    elif model_type == ModelType.LABEL_SELECTION:
        cfg_ = cfg.label_selection_model
        Executor.label_selection_model(
            cfg.batch_size,
            cfg.max_epochs,
            cfg.num_embeddings,
            cfg_.mode,
            cfg_.c,
            savedir,
            datadir,
            seed=cfg.seed,
        )

    else:
        logger.info("Did not execute an experiment or more.")


if __name__ == "__main__":
    main()

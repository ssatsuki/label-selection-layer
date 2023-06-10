from typing import Any, List

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger


def do_experiment(
    data_module,
    model,
    experiment_name: str,
    max_epochs: int,
    logger_type: str = "wandb",
    wandb_project: str = "lightning_logs",
    monitor: str = "Loss/Validation",
    save_top_k: int = 1,
    every_n_epochs: int = 1,
    log_every_n_steps: int = 5,
    should_stop_early: bool = False,
    fast_dev_run: bool = False,
):
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        accelerator_config = {"accelerator": "gpu", "devices": 1}
    else:
        device = torch.device("cpu")
        accelerator_config = {"accelerator": "cpu"}

    callbacks: List[Any] = []

    # prepare checkpoint
    checkpoint = ModelCheckpoint(
        monitor=monitor,
        save_top_k=save_top_k,
        every_n_epochs=every_n_epochs,
    )
    callbacks.append(checkpoint)

    # prepare early stopping
    if should_stop_early:
        early_stop_callback = EarlyStopping(
            monitor=monitor,
            min_delta=0.0,
            patience=5,
            verbose=False,
            mode="min",
        )
        callbacks.append(early_stop_callback)

    # prepare logger
    if logger_type == "wandb":
        logger = WandbLogger(name=experiment_name, save_dir="./logs", project=wandb_project)

    elif logger_type == "tb":
        logger = TensorBoardLogger("./logs", name=experiment_name)  # type: ignore

    elif logger_type == "none":
        logger = None

    else:
        raise ValueError(f"Unsupported logger type: {logger_type}.")

    # prepare data_module on fit stage
    data_module.setup()

    # prepare model
    model = model.to(device)

    # prepare trainer
    if fast_dev_run:
        trainer = pl.Trainer(fast_dev_run=fast_dev_run, **accelerator_config)  # type: ignore
    else:
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            logger=logger,
            log_every_n_steps=log_every_n_steps,
            callbacks=callbacks,
            num_sanity_val_steps=0,
            **accelerator_config,  # type: ignore
        )

    # train model
    trainer.fit(model, data_module)

    # memory efficient
    del data_module.train_data
    del data_module.valid_data

    # predict
    preds = trainer.predict(model, data_module)

    return preds

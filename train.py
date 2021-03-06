import argparse
import logging
import os
import comet_ml  # import required
import torch

from config.default import get_cfg_from_file
from train_utils import (
    get_loss,
    get_optimizer,
    save_checkpoint,
    load_checkpoint,
    training_step,
    model_validation,
    get_lr_scheduler,
    update_scheduler,
    validate_metrics,
)
from dataset import get_dataloader
from utils.comet import init_comet_logging, log_metrics_comet
from utils.logger import init_log
from utils.utilities import get_single_dataloader, is_intersection_empty
from models import get_model


def parser():
    """Parse the arguments."""
    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument(
        "--cfg",
        dest="cfg_path",
        help="Path to the config file",
        type=str,
        default="config/firstrun_focal.yml",
    )
    return parser.parse_args()


def run_training(cfg_path: str) -> None:
    """Runs training for the model specified in the config file.
    Args:
        cfg_path (str): Path to the config file.
    """

    init_log("global", "info")
    logger = logging.getLogger("global")

    cfg = get_cfg_from_file(cfg_path)
    experiment = init_comet_logging(cfg_path)

    logger.info("CONFIG:\n" + str(cfg) + "\n" * 3)

    cfg_name = os.path.splitext(os.path.split(cfg_path)[-1])[0]

    if cfg.TRAIN.WORKERS > 0:
        torch.multiprocessing.set_start_method("spawn", force=True)

    # Load Dataloaders
    train_dataloader = get_dataloader(cfg, "train")
    val_dataloader = get_dataloader(cfg, "val")
    _ = get_dataloader(cfg, "test")

    if not cfg.IS_TEST:
        assert is_intersection_empty(train_dataloader, val_dataloader)

    # load the model
    model = get_model(cfg, cfg.TRAIN.DEVICE)

    # load the optimizer
    optimizer = get_optimizer(model, cfg)

    # load the weights if training resumed
    if os.path.isfile(cfg.TRAIN.RESUME_CHECKPOINT):
        (
            start_epoch,
            weights,
            optimizer_state,
            current_loss,
            checkpoint_cfg,
        ) = load_checkpoint(cfg.TRAIN.RESUME_CHECKPOINT, cfg.TRAIN.DEVICE)
        if checkpoint_cfg != cfg:
            raise Exception("The checkpoint config is different from the config file.")
        model.load_state_dict(weights)
        optimizer.load_state_dict(optimizer_state)
        logger.info(f"Checkpoint {cfg.TRAIN.RESUME_CHECKPOINT} loaded")
    else:
        start_epoch = 1

    criterion = get_loss(cfg)

    epochs = cfg.TRAIN.EPOCHS
    scheduler = get_lr_scheduler(optimizer, cfg, start_epoch - 1)

    # run the training loop
    losses = []
    best_val_metrics = {}
    current_loss = None
    for epoch in range(start_epoch, epochs + 1):
        batch_no = 0
        for train_phase in range(cfg.TRAIN.VAL_PER_EPOCH):
            train_dataloader_single = get_single_dataloader(
                train_dataloader,
                cfg,
                train_phase,
                cfg.TRAIN.VAL_PER_EPOCH,
            )

            for batch in train_dataloader_single:

                # Train step
                batch_no += 1
                loss = training_step(model, optimizer, criterion, batch)
                losses.append(loss.cpu().item())

                if (batch_no + 1) % cfg.TRAIN.VERBOSE_STEP == 0:
                    current_loss = sum(losses) / len(losses)
                    losses = []
                    logger.info(
                        f"Train loss epoch {epoch} "
                        + f"batch {batch_no + 1}: {current_loss:.4f}"
                    )
                    log_metrics_comet(
                        cfg,
                        {
                            "train_loss": current_loss,
                            "learning_rate": optimizer.param_groups[0]["lr"],
                        },
                        experiment,
                        epoch,
                        batch_no,
                    )

            # validation step
            val_metrics = model_validation(model, criterion, val_dataloader)
            val_loss = val_metrics["val_loss"]
            logger.info(f"Val loss at epoch {epoch} batch {batch_no+1}: {val_loss:.4f}")
            update_scheduler(cfg, scheduler, val_loss)
            log_metrics_comet(cfg, val_metrics, experiment, epoch, batch_no)
            validate_metrics(
                val_metrics,
                best_val_metrics,
                cfg_path,
                model,
                epoch,
                optimizer,
                current_loss,
            )

        # save the weight
        logger.info(f"Saving checkpoint at the end of epoch {epoch}")
        save_path = os.path.join(
            cfg.TRAIN.WEIGHTS_FOLDER, f"cfg_{cfg_name}_epoch_{epoch}.pth"
        )
        save_checkpoint(model, epoch, optimizer, current_loss, cfg, save_path)


if __name__ == "__main__":
    args = parser()
    run_training(args.cfg_path)

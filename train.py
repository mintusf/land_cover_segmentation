import os
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
)
from dataset import get_dataloader
from models import get_model


def run_training(cfg_path: str) -> None:
    """Runs training for the model specified in the config file.
    Args:
        cfg_path (str): Path to the config file.
    """

    cfg = get_cfg_from_file(cfg_path)

    cfg_name = os.path.splitext(os.path.split(cfg_path)[-1])[0]

    # Load Dataloaders
    train_dataloader = get_dataloader(cfg, "train")
    val_dataloader = get_dataloader(cfg, "val")

    # load the model
    model = get_model(cfg)

    # load the optimizer
    optimizer = get_optimizer(model, cfg)
    scheduler = get_lr_scheduler(optimizer, cfg)

    # load the weights if training resumed
    if os.path.isfile(cfg.TRAIN.RESUME_CHECKPOINT):
        (
            start_epoch,
            weights,
            optimizer_state,
            current_loss,
            checkpoint_cfg,
        ) = load_checkpoint(cfg, model)
        if checkpoint_cfg != cfg:
            raise Exception("The checkpoint config is different from the config file.")
        model.load_state_dict(optimizer_state)
        optimizer.load_state_dict(weights)
    else:
        start_epoch = 1
        criterion = get_loss(cfg)

    epochs = cfg.TRAIN.EPOCHS

    # run the training loop
    losses = []
    for epoch in range(start_epoch, epochs + 1):
        for i, batch in enumerate(train_dataloader):

            # Train step
            loss = training_step(model, optimizer, criterion, batch)
            losses.append(loss.cpu().item())

            if i % cfg.TRAIN.VERBOSE_STEP == 0:
                current_loss = sum(losses) / len(losses)
                losses = []
                print(f"Training loss at {i + 1} batch: {current_loss}")

            # Val step if N batches passes
            if i % cfg.TRAIN.VAL_STEP == 0:
                # validation step
                val_loss = model_validation(model, criterion, val_dataloader)
                print(f"Validation loss at {i + 1} batch: {val_loss}")
                scheduler.step(val_loss)
                if i == 0:
                    best_val_loss = val_loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_path = os.path.join(
                        cfg.TRAIN.WEIGHTS_FOLDER,
                        f"cfg_{cfg_name}_bestloss.pth",
                    )
                    print("Saving checkpoint")
                    save_checkpoint(
                        model, epoch, optimizer, current_loss, cfg, save_path
                    )

        # save the weight
        save_path = os.path.join(
            cfg.TRAIN.WEIGHTS_FOLDER, f"cfg_{cfg_name}_epoch_{epoch}.pth"
        )
        print("Saving checkpoint")
        save_checkpoint(model, epoch, optimizer, current_loss, cfg, save_path)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn") # If multiple workers
    run_training("config/firstrun_focal.yml")

from train_utils.training_loop import training_loop
from train_utils.losses import get_loss
from train_utils.optimizers import get_optimizer


def run_training(cfg):
    pass
    # set the device to run on
    # device = cfg.TRAIN.DEVICE
    # print("Running on device: {}".format(device))

    # load the data
    # transform = get_transforms(cfg)
    # transforms = Compose([transform])
    # dataset = PatchDataset(cfg, transforms)

    # load the model
    # model = cfg.load_model()
    # model.to(device)

    # load the optimizer
    # optimizer = get_optimizer(cfg, model)

    # load the criterion
    # criterion = get_loss(cfg)

    # run the training loop

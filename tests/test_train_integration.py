import os
from shutil import rmtree

from train import run_training


def test_train_integration(test_config):
    os.mkdir(test_config.TRAIN.WEIGHTS_FOLDER)
    run_training("config/tests.yml")

    rmtree(test_config.TRAIN.WEIGHTS_FOLDER)

import os
from shutil import rmtree

from train import run_training


def test_train_integration(test_config):
    os.makedirs(test_config.TRAIN.WEIGHTS_FOLDER, exist_ok=True)
    run_training("config/tests.yml")

    rmtree(test_config.TRAIN.WEIGHTS_FOLDER)

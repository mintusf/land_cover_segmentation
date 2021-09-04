from test import run_testings


def test_train_integration(module_dict, test_checkpoint):

    run_testings(
        module_dict["cfg_path"],
        test_checkpoint["path"],
        "tests/test_masks",
        add_alphablend=True,
    )

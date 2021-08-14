from dataset import PatchDataset


def test_dataset_init(test_config):

    dataset = PatchDataset(test_config, mode="train")

    assert len(dataset) == 2

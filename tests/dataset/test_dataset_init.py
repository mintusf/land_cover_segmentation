from dataset import PatchDataset


def test_dataset_init(test_config):

    dataset = PatchDataset(test_config, samples_list="train")

    assert len(dataset) == 2

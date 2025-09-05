from .eegfmri_vu import EEGfMRIVuDataset, EEGfMRIVuDatasetConfig
from .NIH_ecr import NIHECRDataset, NIHECRDatasetConfig
from .NIH_ect import NIHECTDataset, NIHECTDatasetConfig

DATASET_FUNCTIONS = {
    "eegfmri_vu": [EEGfMRIVuDataset, EEGfMRIVuDatasetConfig],
    "NIH_ecr": [NIHECRDataset, NIHECRDatasetConfig],
    "NIH_ect": [NIHECTDataset, NIHECTDatasetConfig],
}

def build_dataset(args):
    dataset_builder = DATASET_FUNCTIONS[args.dataset_name][0]
    dataset_config = DATASET_FUNCTIONS[args.dataset_name][1]()
    dataset_dict = {
        "train": dataset_builder(
            dataset_config,
            split_set="train",
        ),
        "test": dataset_builder(
            dataset_config,
            split_set="test",
        ),
        "zero_shot": dataset_builder(
            dataset_config,
            split_set="zero_shot",
        ),
    }
    return dataset_dict, dataset_config

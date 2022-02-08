"""Main interface to preprocess different datasets"""
from omegaconf import OmegaConf

from seiz_eeg.tusz.main import main as tusz_main


if __name__ == "__main__":
    config = OmegaConf.load("data_config.yaml")
    config.merge_with_cli()

    if config.dataset == "tusz":
        tusz_main(config)
    else:
        raise NotImplementedError

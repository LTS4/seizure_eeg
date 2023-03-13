"""Main interface to preprocess different datasets"""
import logging
from typing import Optional

import click
from omegaconf import OmegaConf

from seiz_eeg.preprocess.chbmit.main import main as chbmit_main
from seiz_eeg.preprocess.tusz.main import main as tusz_main


@click.command(
    context_settings=dict(
        allow_extra_args=True,
    )
)
@click.option(
    "-c",
    "--config-path",
    default=None,
    help="Path to configuration file, by default looks for config.yaml.",
    type=click.Path(),
)
def main(config_path: Optional[str] = None):
    """Run preprocessing with options specified in config and cli"""
    if config_path:
        config = OmegaConf.load(config_path)
        config.merge_with_cli()
    else:
        config = OmegaConf.from_cli()

    logging.basicConfig(
        level=config.get("log_level", logging.INFO),
    )

    if config.dataset == "tusz":
        tusz_main(config)
    elif config.dataset == "chbmit":
        chbmit_main(config)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()

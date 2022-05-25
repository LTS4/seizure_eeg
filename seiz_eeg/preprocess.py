"""Main interface to preprocess different datasets"""
import click
from omegaconf import OmegaConf

from seiz_eeg.tusz.main import main as tusz_main


@click.command(
    context_settings=dict(
        allow_extra_args=True,
    )
)
@click.option(
    "-c",
    "--config-path",
    default="config.yaml",
    help="Path to configuration file, by default looks for config.yaml.",
    type=click.Path(exists=True),
)
def main(config_path: str):
    """Run preprocessing with options specified in config and cli"""
    config = OmegaConf.load(config_path)
    config.merge_with_cli()

    if config.dataset == "tusz":
        tusz_main(config)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()

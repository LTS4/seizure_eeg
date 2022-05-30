"""Tests for :module:`seiz_eeg.dataset`"""
# pylint: disable=[redefined-outer-name, no-self-use, pointless-statement]
import pandas as pd
import pytest
from numpy.random import default_rng

from seiz_eeg.dataset import EEGDataset


@pytest.fixture
def segments_df():
    return pd.read_parquet(
        "/home/cappelle/seizure_learning/data/processed/TUSZ/dev/segments.parquet"
    )


class TestEEGDataset:
    """Tests for :py:`EEGDataset`"""

    @pytest.fixture(params=[True, False], ids=lambda x: f"diff_channels:{x}")
    def diff_channels(self, request):
        return request.param

    @pytest.fixture
    def dataset(self, segments_df, diff_channels):
        return EEGDataset(
            segments_df,
            clip_length=10,
            clip_stride=5,
            diff_channels=diff_channels,
            node_level=False,
        )

    def test_getitem(self, dataset: EEGDataset):
        """Test construction and iteration on random samples"""
        rng = default_rng(42)

        for i in rng.integers(len(dataset), size=5):
            dataset[i]

    def test_get_channel_names(self, dataset: EEGDataset):
        dataset.get_channels_names()

    def test_get_label_array(self, dataset: EEGDataset):
        dataset.get_label_array()

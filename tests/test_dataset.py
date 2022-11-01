"""Tests for :module:`seiz_eeg.dataset`"""
# pylint: disable=[redefined-outer-name, no-self-use, pointless-statement]
import pandas as pd
import pytest
from numpy.random import default_rng

from seiz_eeg.clips import make_clips
from seiz_eeg.dataset import EEGDataset


@pytest.fixture
def clips_df():
    return make_clips(
        pd.read_parquet("/home/cappelle/seizure_learning/data/processed/TUSZ/dev/segments.parquet"),
        clip_length=10,
        clip_stride=5,
    )


class TestEEGDataset:
    """Tests for :class:`EEGDataset`"""

    @pytest.fixture(params=[True, False], ids=lambda x: f"diff_channels:{x}")
    def diff_channels(self, request):
        return request.param

    @pytest.fixture
    def dataset(self, clips_df, diff_channels):
        return EEGDataset(
            clips_df,
            diff_channels=diff_channels,
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

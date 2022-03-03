# pylint: disable=[redefined-outer-name, no-self-use, pointless-statement]
import numpy as np
import pandas as pd
import pytest
from numpy.random import default_rng

from seiz_eeg.dataset import EEGDataset, make_clips
from seiz_eeg.schemas import ClipsDF


@pytest.fixture
def segments_df():
    return pd.read_parquet("tests/data/processed/segments.parquet")


@pytest.fixture(params=[60, 33.3], ids=type)
def clip_length(request):
    return request.param


@pytest.fixture(params=[60, 55.5, "start"], ids=type)
def clip_stride(request):
    return request.param


@pytest.fixture
def clips_df_start_stride(segments_df):
    min_len = np.min(segments_df[ClipsDF.end_time] - segments_df[ClipsDF.start_time])
    return make_clips(segments_df, clip_length=min_len, clip_stride="start")


class TestMakeClips:
    """tests for :py:`make_clips`"""

    def test_negative_length_int(self, segments_df: pd.DataFrame):
        assert segments_df.sort_index().equals(make_clips(segments_df, -1, ...))

    def test_negative_length_float(self, segments_df: pd.DataFrame):
        assert segments_df.sort_index().equals(make_clips(segments_df, -0.5, ...))

    def test_lenght_equals_input(self, segments_df: pd.DataFrame, clip_length, clip_stride):
        clips_df = make_clips(segments_df, clip_length, clip_stride)

        assert np.allclose(clips_df[ClipsDF.end_time] - clips_df[ClipsDF.start_time], clip_length)

    def test_negative_stride_raise(self, segments_df):
        with pytest.raises(ValueError):
            make_clips(segments_df, 1, clip_stride=-1)

    def test_invalid_stride_string_raise(self, segments_df):
        with pytest.raises(ValueError):
            make_clips(segments_df, 1, clip_stride="foo")

    def test_start_times_multiples_of_stride(self, segments_df, clip_stride):
        """Test if clips start times are multiples of stride"""
        if isinstance(clip_stride, str):
            return

        clips_df = make_clips(segments_df, clip_stride, clip_stride)
        assert np.allclose(np.diff(clips_df[ClipsDF.start_time]) % clip_stride, 0)

    def test_start_stride_gets_all_segments(self, segments_df, clips_df_start_stride):
        assert len(clips_df_start_stride) == len(segments_df)

    def test_start_stride_start_times(self, segments_df, clips_df_start_stride):
        assert np.allclose(
            clips_df_start_stride[ClipsDF.start_time], segments_df[ClipsDF.start_time].sort_index()
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
            signal_transform=None,
            diff_channels=diff_channels,
            node_level=False,
            device="cpu",
        )

    def test_getitem(self, dataset: EEGDataset):
        """Test construction and iteration on random samples"""
        rng = default_rng(42)

        for i in rng.integers(len(dataset), size=5):
            dataset[i]

    def test_compute_stats(self, dataset: EEGDataset):
        dataset.compute_stats()

    def test_get_channel_names(self, dataset: EEGDataset):
        dataset.get_channels_names()

    def test_get_label_array(self, dataset: EEGDataset):
        dataset.get_label_array()

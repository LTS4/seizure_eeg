"""Tests for :module:`seiz_eeg.clips`"""
# pylint: disable=[redefined-outer-name, no-self-use, pointless-statement]

import numpy as np
import pandas as pd
import pytest

from seiz_eeg.clips import make_clips
from seiz_eeg.schemas import ClipsDF


@pytest.fixture
def segments_df():
    return pd.read_parquet(
        "/home/cappelle/seizure_learning/data/processed/TUSZ/dev/segments.parquet"
    )


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
    """tests for :func:`make_clips`"""

    def test_negative_length_int(self, segments_df: pd.DataFrame):
        assert segments_df.sort_index().equals(make_clips(segments_df, -1, ..., sort_index=True))

    def test_negative_length_float(self, segments_df: pd.DataFrame):
        assert segments_df.sort_index().equals(make_clips(segments_df, -0.5, ..., sort_index=True))

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
            clips_df_start_stride[ClipsDF.start_time].sort_index(),
            segments_df[ClipsDF.start_time].sort_index(),
        )

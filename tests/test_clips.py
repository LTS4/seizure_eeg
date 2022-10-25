"""Tests for :module:`seiz_eeg.clips`"""
# pylint: disable=[redefined-outer-name, no-self-use, pointless-statement]

import numpy as np
import pandas as pd
import pytest
from pandas import IndexSlice as idx

from seiz_eeg.clips import make_clips
from seiz_eeg.schemas import ClipsDF

_PRIMES = [19, 23, 29, 31, 37]


@pytest.fixture
def segments_df():
    """Create sample dataset"""
    schema = ClipsDF.to_schema()
    columns = list(schema.columns.keys())
    index = list(schema.index.columns.keys())
    test_df = pd.DataFrame(columns=index + columns).set_index(index)

    start = 0.0
    end = 0.0
    for i, prime in enumerate(_PRIMES):
        end = start + prime
        test_df.loc["00000000", "00000000_s001_t000", i] = dict(
            label=i % 2,
            start_time=start,
            end_time=end,
            date="2022-10-25",
            sampling_rate=250,
            signals_path="/path/to/signals",
        )
        start = end

    test_df.loc["00000001", "00000001_s001_t000", 0] = dict(
        label=0,
        start_time=0,
        end_time=42,
        date="2022-10-25",
        sampling_rate=250,
        signals_path="/path/to/signals",
    )

    return test_df.astype({key: value.dtype.type for key, value in schema.columns.items()})


@pytest.fixture(params=[5, 15, 30], ids=type)
def clip_length(request):
    return request.param


@pytest.fixture(params=[5, 15, "start"], ids=type)
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

    def test_overlap_ignore(self, segments_df: pd.DataFrame, clip_length: float):
        clips_df = make_clips(
            segments_df, clip_length=clip_length, clip_stride=clip_length, overlap_action="ignore"
        )

        # If the session is not in the index, it means no clip is taken from there
        start_excl = [
            ~np.any(
                sess in clips_df.index.get_level_values(level="session")
                and (
                    (clips_df.loc[idx[pat, sess, :], ClipsDF.start_time] < start_time)
                    & (start_time < clips_df.loc[idx[pat, sess, :], ClipsDF.end_time])
                )
            )
            for (pat, sess, _seg), start_time in segments_df.start_time.items()
        ]
        end_excl = [
            ~np.any(
                sess in clips_df.index.get_level_values(level="session")
                and (
                    (clips_df.loc[idx[pat, sess, :], ClipsDF.start_time] < end_time)
                    & (end_time < clips_df.loc[idx[pat, sess, :], ClipsDF.end_time])
                )
            )
            for (pat, sess, _seg), end_time in segments_df.end_time.items()
        ]

        assert np.all(start_excl) and np.all(end_excl)

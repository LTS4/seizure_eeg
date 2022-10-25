"""Tests for :module:`seiz_eeg.clips`"""
# pylint: disable=[redefined-outer-name, no-self-use, pointless-statement]

import numpy as np
import pandas as pd
import pytest
from pandas import IndexSlice as idx
from pandera.typing import DataFrame

from seiz_eeg.clips import make_clips
from seiz_eeg.schemas import ClipsDF

_PRIMES = [19, 23, 29, 31, 37]


@pytest.fixture
def segments_df() -> DataFrame[ClipsDF]:
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
        end_time=60,
        date="2022-10-25",
        sampling_rate=250,
        signals_path="/path/to/signals",
    )

    return test_df.astype(
        {key: value.dtype.type for key, value in schema.columns.items()}
    ).sort_index()


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


@pytest.fixture(params=["ignore", "left", "right", "seizure", "bkgd"], ids=type)
def overlap_action(request):
    return request.param


class TestMakeClips:
    """tests for :func:`make_clips`"""

    def test_negative_length_int(self, segments_df: DataFrame[ClipsDF]):
        assert segments_df.sort_index().equals(make_clips(segments_df, -1, ..., sort_index=True))

    def test_negative_length_float(self, segments_df: DataFrame[ClipsDF]):
        assert segments_df.sort_index().equals(make_clips(segments_df, -0.5, ..., sort_index=True))

    def test_lenght_equals_input(self, segments_df: DataFrame[ClipsDF], clip_length, clip_stride):
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

    def test_overlap_do_not_overflow(
        self, segments_df: DataFrame[ClipsDF], clip_length: float, overlap_action: str
    ):
        clips_df = make_clips(
            segments_df,
            clip_length=clip_length,
            clip_stride=clip_length,
            overlap_action=overlap_action,
            sort_index=True,
        )

        for (pat, sess), clips_g in clips_df.groupby(level=["patient", "session"]):
            assert (
                clips_g[ClipsDF.end_time].max()
                <= segments_df.loc[idx[pat, sess, :], ClipsDF.end_time].max()
            )

    def test_overlap_ignore(self, segments_df: DataFrame[ClipsDF], clip_length: float):
        clips_df = make_clips(
            segments_df, clip_length=clip_length, clip_stride=clip_length, overlap_action="ignore"
        )

        for (pat, sess, _seg), _label, start_time, end_time, *_ in segments_df.itertuples():
            # If the session is not in the index, it means no clip is taken from there
            if sess in clips_df.index.get_level_values(ClipsDF.session):
                local = clips_df.loc[pat, sess]

                assert ~np.any((local.start_time < start_time) & (start_time < local.end_time))
                assert ~np.any((local.start_time < end_time) & (end_time < local.end_time))

    def test_overlap_left(self, segments_df: DataFrame[ClipsDF], clip_length: float):
        clips_df = make_clips(
            segments_df, clip_length=clip_length, clip_stride=clip_length, overlap_action="left"
        )

        for (pat, sess, _seg), label, _start_time, end_time, *_ in segments_df.itertuples():
            if sess in clips_df.index.get_level_values(ClipsDF.session):
                local = clips_df.loc[pat, sess]
                mask = (local.start_time < end_time) & (end_time <= local.end_time)

                # The last time of a session might be dropped, so the mask can be empty
                assert ~np.any(mask) or local.loc[mask, ClipsDF.label].item() == label

    def test_overlap_right(self, segments_df: DataFrame[ClipsDF], clip_length: float):
        clips_df = make_clips(
            segments_df, clip_length=clip_length, clip_stride=clip_length, overlap_action="right"
        )

        for (pat, sess, _seg), label, start_time, _end_time, *_ in segments_df.itertuples():
            # We canno toverlap 0, so we skip it
            if sess in clips_df.index.get_level_values(ClipsDF.session) and start_time > 0:
                local = clips_df.loc[pat, sess]
                mask = (local.start_time < start_time) & (start_time < local.end_time)

                assert local.loc[mask, ClipsDF.label].item() == label

    def test_overlap_seiz_or_bkgd(self, segments_df: DataFrame[ClipsDF], clip_length: float):
        clips_bkgd = make_clips(
            segments_df, clip_length=clip_length, clip_stride=clip_length, overlap_action="bkgd"
        )

        clips_seiz = make_clips(
            segments_df, clip_length=clip_length, clip_stride=clip_length, overlap_action="seizure"
        )

        for (pat, sess, _seg), _label, start_time, end_time, *_ in segments_df.itertuples():
            # We canno toverlap 0, so we skip it
            if sess in clips_bkgd.index.get_level_values(ClipsDF.session) and start_time > 0:
                local = clips_bkgd.loc[pat, sess]
                mask = (local.start_time < start_time) & (start_time < local.end_time) | (
                    (local.start_time < end_time) & (end_time < local.end_time)
                )

                assert np.all(local.loc[mask, ClipsDF.label] == 0)

            if sess in clips_seiz.index.get_level_values(ClipsDF.session) and start_time > 0:
                local = clips_seiz.loc[pat, sess]
                mask = (local.start_time < start_time) & (start_time < local.end_time) | (
                    (local.start_time < end_time) & (end_time < local.end_time)
                )

                assert np.all(local.loc[mask, ClipsDF.label] > 0)

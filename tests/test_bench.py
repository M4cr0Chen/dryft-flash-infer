"""Regression checks for failures that the old local harness missed."""

from unittest.mock import patch

import pytest

from bench.harness import _check_records, _validate_stream


def test_replay_checks_later_samples():
    records = [(0.1, 0.2, [[i]], [[i + 1]]) for i in range(5)]
    with patch("bench.harness._replay", side_effect=[
        (0.0, True), (0.5, False), (2.5, False), (0.0, True), (0.0, True)
    ]) as replay:
        gap, exact, checks = _check_records(None, "cpu", records, 1)
    assert replay.call_count == 5
    assert gap == 2.5
    assert not exact
    assert checks[2]["tie_gap"] == 2.5


def test_nonfinite_replay_gap_cannot_pass():
    records = [(0.1, 0.2, [[0]], [[1]])] * 2
    with patch("bench.harness._replay", side_effect=[(0.0, True), (float("nan"), False)]):
        gap, _, _ = _check_records(None, "cpu", records, 1)
    assert gap == float("inf")


@pytest.mark.parametrize("emitted", [
    [], [[1, 2]], [[1], [2], [3]], [(1,), (2,)],
    [[True], [2]], [[-1], [2]], [[8], [2]], [[1.0], [2]],
])
def test_invalid_stream_is_rejected(emitted):
    with pytest.raises(ValueError):
        _validate_stream(emitted, batch=1, steps=2, vocab=8)


def test_valid_stream_accepts_eos_as_an_ordinary_id():
    _validate_stream([[0, 7], [0, 7]], batch=2, steps=2, vocab=8)

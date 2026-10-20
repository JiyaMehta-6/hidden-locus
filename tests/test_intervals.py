"""Tests for interval operations."""

import numpy as np

from hidden_locus.core.intervals import (
    IntervalIndex,
    overlap_pairs,
    point_distances,
    point_overlaps,
)


def _index() -> IntervalIndex:
    return IntervalIndex(
        chrom=np.array(["1", "1", "1", "2"], dtype=object),
        start=np.array([100, 200, 300, 50], dtype=np.int64),
        end=np.array([150, 250, 350, 60], dtype=np.int64),
    )


def test_overlap_pairs():
    query = IntervalIndex(
        chrom=np.array(["1", "2"], dtype=object),
        start=np.array([140, 10], dtype=np.int64),
        end=np.array([210, 100], dtype=np.int64),
    )
    q, t = overlap_pairs(query, _index())
    pairs = set(zip(q.tolist(), t.tolist()))
    # query0 [140,210) overlaps intervals 0 [100,150) and 1 [200,250)
    assert (0, 0) in pairs and (0, 1) in pairs
    # query1 [10,100) overlaps interval 3 [50,60)
    assert (1, 3) in pairs
    assert len(pairs) == 3


def test_point_overlaps_inside_boundary():
    idx = _index()
    # 1-based positions: 100 is NOT inside [100,150) because start < pos fails? start=100, pos=100 -> 100 < 100 false.
    p, t = point_overlaps(np.array(["1"], dtype=object), np.array([100]), idx)
    assert len(p) == 0
    # pos=101: start=100 < 101 <= end=150 -> hit
    p, t = point_overlaps(np.array(["1"], dtype=object), np.array([101]), idx)
    assert t.tolist() == [0]
    # pos=150: 100 < 150 <= 150 -> hit (inclusive end)
    p, t = point_overlaps(np.array(["1"], dtype=object), np.array([150]), idx)
    assert t.tolist() == [0]
    # pos=151 -> miss
    p, t = point_overlaps(np.array(["1"], dtype=object), np.array([151]), idx)
    assert len(p) == 0


def test_point_overlaps_multi_hit():
    idx = IntervalIndex(
        chrom=np.array(["1", "1"], dtype=object),
        start=np.array([100, 110], dtype=np.int64),
        end=np.array([200, 210], dtype=np.int64),
    )
    p, t = point_overlaps(np.array(["1"], dtype=object), np.array([150]), idx)
    assert sorted(t.tolist()) == [0, 1]


def test_point_overlaps_chrom_miss():
    p, t = point_overlaps(np.array(["3"], dtype=object), np.array([120]), _index())
    assert len(p) == 0


def test_point_distances_inside_is_zero():
    d = point_distances(np.array(["1", "1"], dtype=object), np.array([120, 149]), _index())
    assert d[0] == 0.0
    assert d[1] == 0.0


def test_point_distances_outside():
    d = point_distances(np.array(["1"], dtype=object), np.array([90]), _index())
    # 0-based point 89; nearest interval starts at 100 -> distance 100 - 89 = 11
    assert d[0] == 11.0
    d = point_distances(np.array(["1"], dtype=object), np.array([260]), _index())
    # 0-based 259; previous interval ends at 250 (exclusive) -> 259 - 250 + 1 = 10
    assert d[0] == 10.0


def test_point_distances_gap_between():
    d = point_distances(np.array(["1"], dtype=object), np.array([175]), _index())
    # between [100,150) and [200,250): left end 149 -> 174-149=25 beats
    # right start 200 -> 200-174=26
    assert d[0] == 25.0


def test_point_distances_missing_chrom_is_inf():
    d = point_distances(np.array(["X"], dtype=object), np.array([120]), _index())
    assert np.isinf(d[0])


def test_overlap_pairs_empty_target():
    empty = IntervalIndex(
        chrom=np.array([], dtype=object),
        start=np.array([], dtype=np.int64),
        end=np.array([], dtype=np.int64),
    )
    q, t = overlap_pairs(_index(), empty)
    assert len(q) == 0

"""Lightweight genomic interval operations (no external interval library).

Intervals are 0-based half-open ``[start, end)`` on single chromosomes.
Chromosome names are handled as opaque strings; callers normalize them
beforehand.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class IntervalIndex:
    """Sorted interval set for one genome (all chromosomes)."""

    chrom: np.ndarray  # object array of chromosome names
    start: np.ndarray  # int64, 0-based inclusive
    end: np.ndarray  # int64, exclusive

    def __post_init__(self) -> None:
        if not (len(self.chrom) == len(self.start) == len(self.end)):
            raise ValueError("IntervalIndex arrays must have equal length")
        if len(self.start) and (self.start < 0).any():
            raise ValueError("Interval starts must be >= 0")

    @classmethod
    def from_frame(cls, frame, chrom_col: str, start_col: str, end_col: str) -> "IntervalIndex":
        return cls(
            chrom=frame[chrom_col].to_numpy(dtype=object),
            start=frame[start_col].to_numpy(dtype=np.int64),
            end=frame[end_col].to_numpy(dtype=np.int64),
        )

    def __len__(self) -> int:
        return len(self.start)


def _order_by_chrom(index: IntervalIndex) -> dict[str, np.ndarray]:
    """Return per-chromosome row indices sorted by start (then by end)."""
    result: dict[str, np.ndarray] = {}
    chrom_arr = index.chrom
    uniq = set(chrom_arr.tolist()) if len(chrom_arr) else set()
    for chrom in uniq:
        rows = np.flatnonzero(chrom_arr == chrom)
        order = np.lexsort((index.end[rows], index.start[rows]))
        result[str(chrom)] = rows[order]
    return result


def overlap_pairs(
    query: IntervalIndex, target: IntervalIndex
) -> tuple[np.ndarray, np.ndarray]:
    """Return (query_row, target_row) index pairs whose intervals overlap."""
    q_rows_by_chrom = _order_by_chrom(query)
    t_rows_by_chrom = _order_by_chrom(target)
    q_out: list[int] = []
    t_out: list[int] = []
    for chrom, q_rows in q_rows_by_chrom.items():
        t_rows = t_rows_by_chrom.get(chrom)
        if t_rows is None or len(t_rows) == 0:
            continue
        t_start = target.start[t_rows]
        t_end = target.end[t_rows]
        # Sweep: add targets whose start is below the query end, drop those
        # whose end is at or before the query start.
        t_ptr = 0
        active: list[tuple[int, int]] = []  # (target end, position in t_rows)
        for q in q_rows:
            q_start = int(query.start[q])
            q_end = int(query.end[q])
            while t_ptr < len(t_rows) and int(t_start[t_ptr]) < q_end:
                heapq.heappush(active, (int(t_end[t_ptr]), t_ptr))
                t_ptr += 1
            while active and active[0][0] <= q_start:
                heapq.heappop(active)
            for _end, pos in active:
                q_out.append(int(q))
                t_out.append(int(t_rows[pos]))
    return np.asarray(q_out, dtype=np.int64), np.asarray(t_out, dtype=np.int64)


def point_overlaps(
    points_chrom: np.ndarray,
    points_pos: np.ndarray,
    target: IntervalIndex,
) -> tuple[np.ndarray, np.ndarray]:
    """Map 1-based point positions to overlapping target intervals.

    Returns ``(point_row, target_row)``; a point inside several intervals
    yields several rows.  A 1-based point ``p`` overlaps the 0-based
    half-open interval ``[start, end)`` iff ``start < p <= end``.
    """
    t_rows_by_chrom = _order_by_chrom(target)
    p_out: list[int] = []
    t_out: list[int] = []
    chrom_arr = np.asarray(points_chrom, dtype=object)
    pos_arr = np.asarray(points_pos, dtype=np.int64)
    for chrom in set(chrom_arr.tolist()) if len(chrom_arr) else set():
        t_rows = t_rows_by_chrom.get(str(chrom))
        if t_rows is None or len(t_rows) == 0:
            continue
        t_start = target.start[t_rows]
        t_end = target.end[t_rows]
        p_rows = np.flatnonzero(chrom_arr == chrom)
        if len(p_rows) == 0:
            continue
        # Sweep: process points by ascending position, maintaining the set of
        # intervals with start < p and end >= p (the active, containing set).
        order = np.argsort(pos_arr[p_rows], kind="mergesort")
        active: list[tuple[int, int]] = []  # (end, position in t_rows)
        t_ptr = 0
        for i in order:
            p = int(p_rows[i])
            pos = int(pos_arr[p])
            while t_ptr < len(t_rows) and int(t_start[t_ptr]) < pos:
                heapq.heappush(active, (int(t_end[t_ptr]), t_ptr))
                t_ptr += 1
            while active and active[0][0] < pos:
                heapq.heappop(active)
            for _end, pos_in_t in active:
                p_out.append(int(p))
                t_out.append(int(t_rows[pos_in_t]))
    return np.asarray(p_out, dtype=np.int64), np.asarray(t_out, dtype=np.int64)


def point_distances(
    points_chrom: np.ndarray,
    points_pos: np.ndarray,
    target: IntervalIndex,
) -> np.ndarray:
    """Distance in bp from each 1-based point to the nearest interval.

    Points inside an interval have distance 0.  Adjacent bases (immediately
    outside) have distance 1.  Chromosomes absent from the target index
    receive ``np.inf``.
    """
    t_rows_by_chrom = _order_by_chrom(target)
    out = np.full(len(points_pos), np.inf, dtype=np.float64)
    chrom_arr = np.asarray(points_chrom, dtype=object)
    pos_arr = np.asarray(points_pos, dtype=np.int64)
    for chrom in set(chrom_arr.tolist()) if len(chrom_arr) else set():
        t_rows = t_rows_by_chrom.get(str(chrom))
        if t_rows is None or len(t_rows) == 0:
            continue
        t_start = target.start[t_rows]
        t_end = target.end[t_rows]
        p_rows = np.flatnonzero(chrom_arr == chrom)
        q = pos_arr[p_rows].astype(np.int64) - 1  # 0-based point coordinate
        # k = last interval index (in start-sorted order) with start <= q.
        k = np.searchsorted(t_start, q, side="right") - 1
        dist = np.full(len(p_rows), np.inf, dtype=np.float64)
        # Distance to the nearest interval starting after the point.
        has_right = k + 1 < len(t_start)
        if has_right.any():
            dist[has_right] = t_start[k[has_right] + 1] - q[has_right]
        # Distance to the nearest interval ending at or before the point.
        has_left = k >= 0
        if has_left.any():
            pref_max_end = np.maximum.accumulate(t_end)
            inside = has_left & (pref_max_end[np.clip(k, 0, None)] > q)
            not_inside = has_left & ~inside
            dist[inside] = 0.0
            if not_inside.any():
                left_dist = q[not_inside] - pref_max_end[k[not_inside]] + 1
                dist[not_inside] = np.minimum(dist[not_inside], left_dist)
        out[p_rows] = dist
    return out


def signed_tss_distance(
    var_chrom: np.ndarray,
    var_pos: np.ndarray,
    gene_chrom: np.ndarray,
    gene_tss: np.ndarray,
    gene_strand: np.ndarray,
) -> np.ndarray:
    """Signed distance from each variant to each gene TSS (broadcast pair-wise).

    Positive when the variant lies downstream of the gene's transcription
    direction, negative when upstream.  Inputs must be equal-length arrays
    describing aligned (variant, gene) pairs.
    """
    delta = var_pos.astype(np.int64) - gene_tss.astype(np.int64)
    strand = np.asarray(gene_strand, dtype=object)
    sign = np.where(strand == "-", -1, 1)
    return delta * sign

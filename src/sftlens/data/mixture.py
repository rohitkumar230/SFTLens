"""Proportional stratified subsampling of a mixture dataset.

The tulu-3 mixture is 939k examples across 19 sources with very uneven shares.
The recipe's learning rate was selected against that composition, so a subset
used as a stand-in has to preserve it. Uniform random sampling preserves the
proportions only in expectation; stratifying makes them exact, which matters
when a small source (tulu_hard_coded, 0.03%) would otherwise vanish or be
over-represented by chance.
"""

from __future__ import annotations

from collections import Counter

import numpy as np


def largest_remainder_quota(shares: dict[str, int], total: int) -> dict[str, int]:
    """Allocate `total` slots across strata in proportion to `shares`.

    Plain rounding does not sum to `total`. The largest-remainder method
    distributes the leftover to the strata with the biggest fractional parts,
    which keeps every allocation within one of its ideal value.
    """
    if total <= 0:
        raise ValueError(f"total must be positive, got {total}")
    pool = sum(shares.values())
    if total > pool:
        raise ValueError(f"requested {total} examples but the pool holds {pool}")

    exact = {k: v * total / pool for k, v in shares.items()}
    quota = {k: int(np.floor(v)) for k, v in exact.items()}

    # Hand out the remainder by descending fractional part; ties break on the
    # stratum name so the allocation is reproducible across runs and platforms.
    leftover = total - sum(quota.values())
    order = sorted(exact, key=lambda k: (-(exact[k] - quota[k]), k))
    for k in order[:leftover]:
        quota[k] += 1

    # A stratum can only give what it has; spill overflow to the others.
    for k, cap in shares.items():
        if quota[k] > cap:
            quota[k] = cap
    deficit = total - sum(quota.values())
    if deficit:
        headroom = sorted((k for k in shares if quota[k] < shares[k]),
                          key=lambda k: (-shares[k], k))
        for k in headroom:
            take = min(deficit, shares[k] - quota[k])
            quota[k] += take
            deficit -= take
            if not deficit:
                break

    return quota


def stratified_indices(
    labels: list, total: int, seed: int
) -> list[int]:
    """Indices of a proportional stratified sample, sorted ascending.

    Returned sorted because `datasets.Dataset.select` on a sorted index list
    reads sequentially from the arrow file instead of seeking per row.
    """
    shares = Counter(labels)
    quota = largest_remainder_quota(dict(shares), total)

    by_stratum: dict[object, list[int]] = {}
    for i, lab in enumerate(labels):
        by_stratum.setdefault(lab, []).append(i)

    rng = np.random.default_rng(seed)
    picked: list[int] = []
    for stratum in sorted(by_stratum, key=str):
        pool = by_stratum[stratum]
        k = quota.get(stratum, 0)
        if k <= 0:
            continue
        take = rng.choice(len(pool), size=k, replace=False)
        picked.extend(pool[j] for j in take)

    return sorted(picked)


def composition(labels: list) -> dict[str, float]:
    """Stratum shares as percentages, for the run manifest."""
    counts = Counter(labels)
    n = sum(counts.values()) or 1
    return {str(k): 100.0 * v / n for k, v in counts.most_common()}


def format_composition(labels: list, title: str = "composition") -> str:
    counts = Counter(labels)
    n = sum(counts.values()) or 1
    lines = [f"[data] {title}: {n} examples across {len(counts)} strata"]
    for k, v in counts.most_common():
        lines.append(f"       {v:8d}  {100 * v / n:5.2f}%  {k}")
    return "\n".join(lines)

"""Stratified subsampling tests.

The subset stands in for the mixture the recipe's learning rate was selected
against, so its composition has to be preserved exactly rather than in
expectation.
"""

from __future__ import annotations

from collections import Counter

import pytest

from sftlens.data.mixture import composition, largest_remainder_quota, stratified_indices

# Real tulu-3-sft-mixture source counts (939,343 examples, 19 sources).
TULU3 = {
    "personahub_math_v5": 149960, "evol_codealpaca": 107276, "wildchat_100k": 100000,
    "aya_100k": 100000, "flan_v2": 89982, "numinamath_tir": 64312,
    "wildguardmix_50k": 50000, "open_math_2_gsm8k_50k": 50000, "wildjailbreak_50k": 50000,
    "personas_math_grade": 49980, "personahub_code_v2": 34999, "personahub_ifdata": 29980,
    "personahub_math_interm_algebra": 20000, "coconot": 10983, "sciriff_10k": 10000,
    "no_robots": 9500, "oasst1": 7131, "table_gpt_5k": 5000, "hard_coded_repeated_10": 240,
}


class TestQuota:
    def test_allocates_exactly_the_requested_total(self):
        for total in (100, 1000, 50_000, 123_457):
            assert sum(largest_remainder_quota(TULU3, total).values()) == total

    def test_every_allocation_is_within_one_of_ideal(self):
        total = 50_000
        pool = sum(TULU3.values())
        quota = largest_remainder_quota(TULU3, total)
        for source, count in TULU3.items():
            ideal = count * total / pool
            assert abs(quota[source] - ideal) < 1.0, source

    def test_smallest_source_survives(self):
        """hard_coded_repeated_10 is 0.026% of the mixture. Uniform sampling
        would sometimes drop it entirely; stratification must not."""
        quota = largest_remainder_quota(TULU3, 50_000)
        assert quota["hard_coded_repeated_10"] >= 12

    def test_never_over_draws_a_stratum(self):
        """A stratum cannot supply more rows than it holds; the deficit must
        spill to strata with headroom rather than produce an impossible quota."""
        shares = {"tiny": 3, "huge": 997}
        quota = largest_remainder_quota(shares, 500)
        assert quota["tiny"] <= 3
        assert sum(quota.values()) == 500

    def test_rejects_a_total_larger_than_the_pool(self):
        with pytest.raises(ValueError, match="pool holds"):
            largest_remainder_quota({"a": 10}, 11)

    def test_rejects_a_non_positive_total(self):
        with pytest.raises(ValueError, match="must be positive"):
            largest_remainder_quota({"a": 10}, 0)

    def test_is_deterministic(self):
        assert largest_remainder_quota(TULU3, 7_777) == largest_remainder_quota(TULU3, 7_777)


class TestStratifiedIndices:
    @pytest.fixture
    def labels(self):
        return [s for s, n in TULU3.items() for _ in range(n // 100)]

    def test_returns_the_requested_count(self, labels):
        assert len(stratified_indices(labels, 2000, seed=0)) == 2000

    def test_preserves_composition_to_within_a_tenth_of_a_percent(self, labels):
        idx = stratified_indices(labels, 2000, seed=0)
        before = composition(labels)
        after = composition([labels[i] for i in idx])
        for source, share in before.items():
            assert abs(after.get(source, 0.0) - share) < 0.1, source

    def test_indices_are_unique_and_sorted(self, labels):
        idx = stratified_indices(labels, 2000, seed=0)
        assert len(set(idx)) == len(idx)
        assert idx == sorted(idx)

    def test_same_seed_gives_the_same_sample(self, labels):
        assert stratified_indices(labels, 500, seed=7) == stratified_indices(labels, 500, seed=7)

    def test_different_seed_gives_a_different_sample(self, labels):
        assert stratified_indices(labels, 500, seed=7) != stratified_indices(labels, 500, seed=8)

    def test_composition_is_seed_independent(self, labels):
        """Which rows are drawn depends on the seed; how many per source
        does not. That is the difference between stratified and uniform."""
        a = Counter(labels[i] for i in stratified_indices(labels, 1500, seed=1))
        b = Counter(labels[i] for i in stratified_indices(labels, 1500, seed=2))
        assert a == b

    def test_single_stratum_degenerates_gracefully(self):
        idx = stratified_indices(["only"] * 100, 30, seed=0)
        assert len(idx) == 30 and len(set(idx)) == 30

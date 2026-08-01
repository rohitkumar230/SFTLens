"""Persistence tests.

These target the failure mode where telemetry is lost or corrupted by a resume
-- which is precisely the run whose telemetry you most want to read.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from sftlens.telemetry.writer import TelemetryWriter, WeightBaseline


@pytest.fixture
def writer(tmp_path):
    return TelemetryWriter(tmp_path, flush_rows=4)


def _rows(steps):
    return [{"step": s, "name": "m", "c": 0.1 * s} for s in steps]


class TestScalarShards:
    def test_flushes_on_the_row_threshold(self, writer):
        writer.add_rows(_rows([1, 2, 3]))
        assert list(writer.scalars_dir.glob("*.parquet")) == []
        writer.add_rows(_rows([4]))
        assert len(list(writer.scalars_dir.glob("*.parquet"))) == 1

    def test_shard_name_encodes_the_step_range(self, writer):
        writer.add_rows(_rows([10, 20, 30, 40]))
        (path,) = writer.scalars_dir.glob("*.parquet")
        assert path.name == "steps_0000010_0000040.parquet"

    def test_a_resumed_run_does_not_overwrite_earlier_shards(self, tmp_path):
        """The original design keyed shards on an in-memory counter, which
        resets on resume; the first post-resume flush then clobbered the shard
        written before the crash."""
        first = TelemetryWriter(tmp_path, flush_rows=100)
        first.add_rows(_rows([1, 2]))
        first.flush()

        resumed = TelemetryWriter(tmp_path, flush_rows=100)   # counter back to zero
        resumed.add_rows(_rows([3, 4]))
        resumed.flush()

        shards = sorted(p.name for p in (tmp_path / "scalars").glob("*.parquet"))
        assert shards == ["steps_0000001_0000002.parquet", "steps_0000003_0000004.parquet"]

    def test_a_re_probed_step_is_kept_not_silently_replaced(self, tmp_path):
        """Re-probing a step after resume must leave both records visible."""
        for _ in range(2):
            w = TelemetryWriter(tmp_path, flush_rows=100)
            w.add_rows(_rows([5, 6]))
            w.flush()

        shards = sorted(p.name for p in (tmp_path / "scalars").glob("*.parquet"))
        assert shards == [
            "steps_0000005_0000006.dup01.parquet",
            "steps_0000005_0000006.parquet",
        ]

    def test_flush_is_idempotent_when_empty(self, writer):
        writer.flush()
        writer.flush()
        assert list(writer.scalars_dir.glob("*.parquet")) == []

    def test_rows_survive_the_round_trip(self, writer):
        import pandas as pd

        writer.add_rows(_rows([1, 2, 3, 4]))
        (path,) = writer.scalars_dir.glob("*.parquet")
        df = pd.read_parquet(path)
        assert len(df) == 4
        assert sorted(df["step"]) == [1, 2, 3, 4]


class TestDeepDumps:
    def test_writes_and_reloads(self, writer):
        payload = {"layer.0|eig_K": np.arange(4, dtype=np.float32)}
        writer.save_deep(120, payload)
        (path,) = writer.deep_dir.glob("*.npz")
        assert path.name == "step_0000120.npz"
        assert np.array_equal(np.load(path)["layer.0|eig_K"], payload["layer.0|eig_K"])

    def test_empty_payload_writes_nothing(self, writer):
        writer.save_deep(10, {})
        assert list(writer.deep_dir.glob("*.npz")) == []


class TestWeightBaseline:
    def _weights(self, scale=1.0):
        torch.manual_seed(0)
        return {"m.0": torch.randn(4, 6) * scale}

    def test_captures_then_reloads_the_same_reference(self, tmp_path):
        w = self._weights()
        first = WeightBaseline(tmp_path / "w0.pt")
        first.capture_or_load(w)
        assert (tmp_path / "w0.pt").exists()

        # Simulate training having moved the weights, then a resume.
        moved = {"m.0": w["m.0"] + 1.0}
        second = WeightBaseline(tmp_path / "w0.pt")
        second.capture_or_load(moved)

        delta = second.delta("m.0", moved["m.0"])
        assert torch.allclose(delta, torch.ones(4, 6), atol=1e-6), (
            "dW must be measured from step 0, not from the resume point"
        )

    def test_delta_from_a_fresh_baseline_is_zero(self, tmp_path):
        w = self._weights()
        b = WeightBaseline(tmp_path / "w0.pt")
        b.capture_or_load(w)
        assert torch.equal(b.delta("m.0", w["m.0"]), torch.zeros(4, 6)), (
            "an fp32 baseline must give exactly zero displacement at step 0; "
            "any floor here propagates into every early dW_relnorm"
        )

    def test_unknown_module_returns_none_rather_than_fabricating(self, tmp_path):
        b = WeightBaseline(tmp_path / "w0.pt")
        b.capture_or_load(self._weights())
        assert b.delta("m.99", torch.randn(4, 6)) is None

    def test_missing_module_after_a_config_change_is_reported(self, tmp_path, capsys):
        WeightBaseline(tmp_path / "w0.pt").capture_or_load({"m.0": torch.randn(2, 2)})

        resumed = WeightBaseline(tmp_path / "w0.pt")
        resumed.capture_or_load({"m.0": torch.randn(2, 2), "m.1": torch.randn(2, 2)})
        assert "missing 1 tracked modules" in capsys.readouterr().out
        assert resumed.delta("m.1", torch.randn(2, 2)) is None

    def test_no_temp_file_is_left_behind(self, tmp_path):
        b = WeightBaseline(tmp_path / "w0.pt")
        b.capture_or_load(self._weights())
        assert list(tmp_path.glob("*.tmp")) == []

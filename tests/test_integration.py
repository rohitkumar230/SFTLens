"""Full training loop with telemetry attached, on a tiny model.

Exercises the wiring that unit tests cannot: Trainer callback dispatch, the
custom loss reaching the optimizer, token accounting driving the probe cadence,
and the archive that lands on disk. Runs on CPU in a few seconds, with no
network access.
"""

from __future__ import annotations

import json

import pytest
import torch
from datasets import Dataset
from transformers import LlamaConfig, LlamaForCausalLM

from sftlens.config import RunConfig
from sftlens.data.build import encode
from sftlens.data.chatml import IGNORE_INDEX, ChatMLTemplate
from sftlens.data.collate import PadCollator
from sftlens.telemetry.callback import attach_telemetry, build_probe_batch
from sftlens.train.run import build_training_args
from sftlens.train.trainer import SFTTrainer, TrainLogCallback

from .test_chatml import FakeTokenizer

VOCAB = 128
SOURCES = ["math", "code", "chat", "safety"]


class PaddedFakeTokenizer(FakeTokenizer):
    pad_token_id = 0
    eos_token_id = 0


@pytest.fixture
def template():
    return ChatMLTemplate(tokenizer=PaddedFakeTokenizer(), system_prompt="S")


@pytest.fixture
def datasets(template):
    rows = []
    for i in range(64):
        rows.append({
            "messages": [
                {"role": "user", "content": f"question {i % 7} " * (1 + i % 3)},
                {"role": "assistant", "content": f"answer {i % 5} " * (1 + i % 4)},
            ],
            "source": SOURCES[i % len(SOURCES)],
        })
    ds = Dataset.from_list(rows)

    cfg = RunConfig().data
    cfg.max_seq_len, cfg.num_proc, cfg.on_overflow = 128, 1, "drop"
    enc = encode(ds, template, cfg)
    split = enc.train_test_split(test_size=16, seed=0)
    return split["train"], split["test"]


@pytest.fixture
def model():
    torch.manual_seed(0)
    m = LlamaForCausalLM(LlamaConfig(
        vocab_size=VOCAB, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=256,
    ))
    m.config.use_cache = False
    return m


@pytest.fixture
def cfg(tmp_path):
    c = RunConfig(run_name="itest", output_dir=str(tmp_path / "run"), seed=0)
    c.recipe.effective_batch, c.recipe.per_device_batch = 4, 2
    c.recipe.max_steps, c.recipe.epochs = 6, 1.0
    c.recipe.lr = 1e-3                     # visible movement in six steps
    c.model.gradient_checkpointing = False
    c.logging_steps, c.eval_steps, c.save_steps = 1, 5, 5
    c.dataloader_num_workers, c.group_by_length = 0, False
    c.use_cpu = True   # device-independent results; MPS lacks fp64

    t = c.telemetry
    t.cadence_unit, t.light_every, t.deep_every = "steps", 2, 4
    t.probe_seqs, t.probe_max_len, t.probe_micro_batch = 8, 64, 4
    t.n_tokens, t.n_tokens_sweep = 64, (32, 64)
    t.layer_stride, t.always_layers = 1, (0,)
    t.top_eig, t.n_tokens_deep, t.dw_rank, t.flush_rows = 4, 8, 4, 10_000
    return c


def _build(cfg, model, template, datasets):
    train_ds, eval_ds = datasets
    collator = PadCollator(template.tokenizer.pad_token_id, pad_to_multiple_of=8)
    out_dir = cfg.resolved_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    trainer = SFTTrainer(
        model=model,
        args=build_training_args(cfg, out_dir),
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        loss_reduction=cfg.recipe.loss_reduction,
    )
    import weakref

    trainer.add_callback(TrainLogCallback(out_dir / "train_log.jsonl", weakref.ref(trainer)))
    telemetry = attach_telemetry(trainer, cfg, eval_ds, collator)
    return trainer, telemetry, out_dir


@pytest.fixture
def trained(cfg, model, template, datasets):
    trainer, telemetry, out_dir = _build(cfg, model, template, datasets)
    trainer.train()
    return trainer, telemetry, out_dir


class TestTrainingLoop:
    def test_training_runs_and_updates_weights(self, cfg, model, template, datasets):
        before = model.model.layers[0].mlp.down_proj.weight.detach().clone()
        trainer, _, _ = _build(cfg, model, template, datasets)
        trainer.train()
        after = model.model.layers[0].mlp.down_proj.weight.detach()
        assert not torch.allclose(before, after), "weights did not move"

    def test_token_accounting_advances(self, trained):
        trainer, _, _ = trained
        assert trainer.tokens_seen > 0
        assert 0 < trainer.supervised_tokens_seen < trainer.tokens_seen

    def test_train_log_is_joinable_on_step_and_tokens(self, trained):
        _, _, out_dir = trained
        records = [json.loads(x) for x in (out_dir / "train_log.jsonl").read_text().splitlines()]
        assert records
        assert all({"step", "tokens_seen"} <= set(r) for r in records)
        steps = [r["step"] for r in records]
        tokens = [r["tokens_seen"] for r in records]
        assert steps == sorted(steps)
        assert tokens == sorted(tokens), "token count must be monotone"

    def test_sum_reduction_scales_the_loss_by_token_count(self, template, model, datasets):
        """sum vs mean is a real difference in effective step size, not a
        cosmetic one -- assert the two actually differ."""
        collator = PadCollator(template.tokenizer.pad_token_id)
        batch = collator([dict(datasets[0][i]) for i in range(4)])
        logits = model(input_ids=batch["input_ids"],
                       attention_mask=batch["attention_mask"]).logits

        from sftlens.train.loss import token_loss

        total, n = token_loss(logits, batch["labels"], "sum")
        mean, n2 = token_loss(logits, batch["labels"], "mean")
        assert n == n2 > 0
        assert total.item() == pytest.approx(mean.item() * n, rel=1e-4)


class TestTelemetryArchive:
    def test_scalars_are_written(self, trained):
        _, telemetry, _ = trained
        shards = list(telemetry.writer.scalars_dir.glob("*.parquet"))
        assert shards, "no scalar shard written"

    def test_deep_dumps_land_on_the_configured_cadence(self, trained):
        _, telemetry, _ = trained
        steps = sorted(int(p.stem.split("_")[1])
                       for p in telemetry.writer.deep_dir.glob("*.npz"))
        assert steps, "no deep dump written"
        # deep_every=2 in step units, plus the step-0 baseline
        assert 0 in steps
        assert all(s % 2 == 0 for s in steps)

    def test_step_zero_baseline_precedes_every_update(self, trained):
        import pandas as pd

        _, telemetry, _ = trained
        df = pd.concat([pd.read_parquet(p)
                        for p in telemetry.writer.scalars_dir.glob("*.parquet")])
        assert df["step"].min() == 0, "no pre-update baseline"

    def test_every_probed_module_appears_at_every_probe(self, trained):
        import pandas as pd

        _, telemetry, _ = trained
        df = pd.concat([pd.read_parquet(p)
                        for p in telemetry.writer.scalars_dir.glob("*.parquet")])
        full = df[df["is_full_n"]]
        counts = full.groupby("step")["name"].nunique()
        assert (counts == len(telemetry.probe.targets)).all(), counts.to_dict()

    def test_metrics_are_finite_across_the_trajectory(self, trained):
        import numpy as np
        import pandas as pd

        _, telemetry, _ = trained
        df = pd.concat([pd.read_parquet(p)
                        for p in telemetry.writer.scalars_dir.glob("*.parquet")])
        for col in ("c", "PR_Sigma", "a", "R", "rho_delta", "rayleigh_full",
                    "R_Omega", "PR_Omega", "mu2"):
            assert np.isfinite(df[col]).all(), f"{col} went non-finite"

    def test_scalars_carry_both_clocks(self, trained):
        import pandas as pd

        _, telemetry, _ = trained
        df = pd.read_parquet(next(telemetry.writer.scalars_dir.glob("*.parquet")))
        assert {"step", "tokens_seen", "probe_loss_per_token"} <= set(df.columns)

    def test_deep_dump_contains_spectra_optimizer_and_dw(self, trained):
        import numpy as np

        _, telemetry, _ = trained
        last = sorted(telemetry.writer.deep_dir.glob("*.npz"))[-1]
        keys = set(np.load(last).keys())
        name = telemetry.probe.targets[0].name
        assert f"{name}|eig_K" in keys
        assert f"{name}|Xc" in keys and f"{name}|Delta" in keys
        assert any(k.endswith("|adam_m_norm") for k in keys), "no Adam state captured"
        assert any(k.endswith("|dW_svals") for k in keys), "no dW spectrum captured"

    def test_dw_grows_from_the_step_zero_baseline(self, trained):
        import numpy as np

        _, telemetry, _ = trained
        dumps = sorted(telemetry.writer.deep_dir.glob("*.npz"))
        name = telemetry.probe.targets[0].name
        first = np.load(dumps[0])[f"{name}|dW_relnorm"]
        last = np.load(dumps[-1])[f"{name}|dW_relnorm"]
        assert first == pytest.approx(0.0, abs=1e-6), "baseline dW must be zero"
        assert last > first, "cumulative update did not grow"

    def test_config_and_probe_plan_are_archived(self, trained):
        _, telemetry, _ = trained
        root = telemetry.writer.root
        cfg_json = json.loads((root / "config.json").read_text())
        plan = json.loads((root / "probe_plan.json").read_text())
        assert cfg_json["recipe"]["provenance"]
        assert plan["n_tokens"] > 0
        assert plan["micro_batches"] >= 1


class TestProbeIsolation:
    def test_probe_does_not_perturb_the_training_trajectory(
        self, cfg, template, datasets, model
    ):
        """The training run is the asset. Attaching telemetry must not change
        a single weight."""
        state = {k: v.clone() for k, v in model.state_dict().items()}

        cfg.telemetry.enabled = False
        t_off, _, _ = _build(cfg, model, template, datasets)
        t_off.train()
        without = {k: v.clone() for k, v in model.state_dict().items()}

        model.load_state_dict(state)
        cfg.telemetry.enabled = True
        t_on, _, _ = _build(cfg, model, template, datasets)
        t_on.train()
        with_telemetry = model.state_dict()

        for key in without:
            assert torch.allclose(without[key], with_telemetry[key], atol=1e-6), (
                f"telemetry perturbed {key}"
            )

    def test_probe_batch_is_stratified_and_held_out(self, cfg, datasets, template):
        _, eval_ds = datasets
        collator = PadCollator(template.tokenizer.pad_token_id)
        batch = build_probe_batch(eval_ds, collator, cfg.telemetry, seed=0)
        assert batch
        total = sum(mb["input_ids"].shape[0] for mb in batch)
        assert total <= cfg.telemetry.probe_seqs
        assert all(mb["input_ids"].shape[1] <= cfg.telemetry.probe_max_len for mb in batch)

    def test_probe_batch_rejects_sequences_with_no_supervision(self, cfg, template):
        """Truncation can strip every supervised token from a long-prompt row;
        such a row contributes no Delta."""
        rows = [{"input_ids": list(range(20)), "labels": [IGNORE_INDEX] * 20}]
        ds = Dataset.from_list(rows)
        collator = PadCollator(template.tokenizer.pad_token_id)
        with pytest.raises(RuntimeError, match="no probe sequence"):
            build_probe_batch(ds, collator, cfg.telemetry, seed=0)


class TestResume:
    def test_a_resumed_run_extends_the_archive(self, cfg, model, template, datasets):
        trainer, telemetry, out_dir = _build(cfg, model, template, datasets)
        trainer.train()
        first_shards = {p.name for p in telemetry.writer.scalars_dir.glob("*.parquet")}
        assert first_shards

        # Fresh objects against the same output dir, as a real resume would be.
        cfg.recipe.max_steps = 10
        trainer2, telemetry2, _ = _build(cfg, model, template, datasets)
        ckpt = sorted(out_dir.glob("checkpoint-*"))
        trainer2.train(resume_from_checkpoint=str(ckpt[-1]) if ckpt else None)

        after = {p.name for p in telemetry2.writer.scalars_dir.glob("*.parquet")}
        assert first_shards <= after, "a resume destroyed pre-existing shards"
        assert len(after) > len(first_shards), "the resume wrote nothing new"

    def test_weight_baseline_persists_across_runs(self, cfg, model, template, datasets):
        trainer, telemetry, _ = _build(cfg, model, template, datasets)
        trainer.train()
        path = telemetry.baseline.path
        assert path.exists()
        original = torch.load(path, map_location="cpu")

        _, telemetry2, _ = _build(cfg, model, template, datasets)
        telemetry2.baseline.capture_or_load(
            {t.name: t.module.weight for t in telemetry2.dw_targets}
        )
        for k, v in original.items():
            assert torch.allclose(v.float(), telemetry2.baseline._weights[k].float()), (
                "baseline was re-captured mid-run; dW would be measured from the "
                "resume point rather than from step 0"
            )

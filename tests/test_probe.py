"""End-to-end probe tests against a real (tiny) Llama model.

SmolLM2 is a LlamaForCausalLM, so a randomly initialised tiny Llama exercises
the same module naming, the same hook plumbing, and the same code path as the
production run -- in a second, on CPU.
"""

from __future__ import annotations

import math

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from sftlens.config import TelemetryConfig
from sftlens.data.chatml import IGNORE_INDEX
from sftlens.telemetry.probe import GramProbe, select_targets
from sftlens.train.loss import token_loss

HIDDEN, INTERMEDIATE, LAYERS = 32, 96, 4


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=128, hidden_size=HIDDEN, intermediate_size=INTERMEDIATE,
        num_hidden_layers=LAYERS, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=64,
    )
    m = LlamaForCausalLM(cfg)
    m.config.use_cache = False
    return m


@pytest.fixture
def cfg():
    return TelemetryConfig(
        n_tokens=48, n_tokens_sweep=(16, 32, 48), sweep_on_deep_only=True,
        layer_stride=2, always_layers=(0,), top_eig=4, n_tokens_deep=8,
        probe_seqs=4, probe_max_len=16, seed=99,
    )


@pytest.fixture
def batch():
    torch.manual_seed(1)
    def make(bs, seq):
        ids = torch.randint(3, 128, (bs, seq))
        labels = ids.clone()
        labels[:, : seq // 2] = IGNORE_INDEX      # mask a prompt span
        return {"input_ids": ids, "labels": labels,
                "attention_mask": torch.ones(bs, seq, dtype=torch.long)}
    return [make(2, 16), make(2, 16)]             # two micro-batches


def loss_fn(logits, labels):
    return token_loss(logits, labels, "sum")


def test_targets_cover_expected_modules(model, cfg):
    targets = select_targets(model, cfg)
    layers = sorted({t.layer for t in targets})
    assert layers == [0, 2, 3]                    # stride 2, plus always and last
    assert len({t.suffix for t in targets}) == 7
    assert all(isinstance(t.module, torch.nn.Linear) for t in targets)


def test_probe_produces_finite_metrics_for_every_module(model, cfg, batch):
    probe = GramProbe(model, cfg)
    probe.plan(batch)
    probe.run(batch, loss_fn)
    rows, artifacts = probe.reduce(deep=False)

    assert len(rows) == len(probe.targets), "a module produced no row"
    assert artifacts == {}
    for r in rows:
        for key in ("c", "PR_Sigma", "a", "R", "rho_delta", "rayleigh_full", "PR_Omega"):
            assert math.isfinite(r[key]), f"{r['name']}.{key} = {r[key]}"
        assert r["g2"] > 0


def test_down_proj_input_dimension_is_the_intermediate_size(model, cfg, batch):
    """The dimension contrast the study rests on must actually be present."""
    probe = GramProbe(model, cfg)
    probe.plan(batch)
    probe.run(batch, loss_fn)
    rows, _ = probe.reduce(deep=False)

    by_module = {r["module"]: r for r in rows if r["layer"] == 0}
    assert by_module["mlp.down_proj"]["D_in"] == INTERMEDIATE
    assert by_module["self_attn.o_proj"]["D_in"] == HIDDEN
    # D_out held equal: this is the controlled pair.
    assert by_module["mlp.down_proj"]["D_out"] == by_module["self_attn.o_proj"]["D_out"]


def test_probe_leaves_no_parameter_gradients(model, cfg, batch):
    """The probe must not touch the optimizer's view of the model.

    Parameters are frozen and the graph is rooted at the embeddings, so no
    .grad buffer should exist afterwards.
    """
    model.zero_grad(set_to_none=True)
    probe = GramProbe(model, cfg)
    probe.plan(batch)
    probe.run(batch, loss_fn)

    with_grad = [n for n, p in model.named_parameters() if p.grad is not None]
    assert with_grad == [], f"probe allocated gradients for {with_grad[:3]}"


def test_probe_restores_requires_grad_and_mode(model, cfg, batch):
    model.train()
    before = {n: p.requires_grad for n, p in model.named_parameters()}

    probe = GramProbe(model, cfg)
    probe.plan(batch)
    probe.run(batch, loss_fn)

    assert model.training, "probe left the model in eval mode"
    assert {n: p.requires_grad for n, p in model.named_parameters()} == before


def test_hooks_are_removed_after_the_probe(model, cfg, batch):
    probe = GramProbe(model, cfg)
    probe.plan(batch)
    probe.run(batch, loss_fn)

    for t in probe.targets:
        assert not t.module._forward_hooks, f"{t.name} kept a forward hook"
        assert not t.module._backward_hooks, f"{t.name} kept a backward hook"


def test_token_selection_is_identical_across_probes(model, cfg, batch):
    """The whole point of a longitudinal measurement: the same tokens of the
    same sequences are measured at every checkpoint, so step-to-step variation
    is network change and not resampling noise."""
    probe = GramProbe(model, cfg)
    plan_a = probe.plan(batch)
    gather_a = [plan_a[i]["gather"].clone() for i in range(len(batch))]

    plan_b = probe.plan(batch)
    for i in range(len(batch)):
        assert torch.equal(gather_a[i], plan_b[i]["gather"])


def test_identical_model_state_gives_identical_metrics(model, cfg, batch):
    """Two probes with no update between them must agree exactly. Any drift is
    resampling noise leaking into the trajectory."""
    probe = GramProbe(model, cfg)
    probe.plan(batch)

    probe.run(batch, loss_fn)
    first, _ = probe.reduce(deep=False)
    probe.clear()
    probe.run(batch, loss_fn)
    second, _ = probe.reduce(deep=False)

    for a, b in zip(first, second, strict=True):
        assert a["name"] == b["name"]
        for key in ("c", "a", "R", "g2", "PR_Omega"):
            assert a[key] == pytest.approx(b[key], rel=1e-6), f"{a['name']}.{key} drifted"


def test_sweep_emits_a_row_per_n_only_on_deep(model, cfg, batch):
    probe = GramProbe(model, cfg)
    probe.plan(batch)
    probe.run(batch, loss_fn)

    light_rows, _ = probe.reduce(deep=False)
    assert {r["N"] for r in light_rows} == {48}
    assert all(r["is_full_n"] for r in light_rows)

    deep_rows, _ = probe.reduce(deep=True)
    assert {r["N"] for r in deep_rows} == {16, 32, 48}
    assert sum(r["is_full_n"] for r in deep_rows) == len(probe.targets)


def test_pr_grows_with_n_in_the_sweep(model, cfg, batch):
    """The finite-N floor must be visible in the emitted data: measured PR
    increases with N even though the underlying covariance is unchanged."""
    probe = GramProbe(model, cfg)
    probe.plan(batch)
    probe.run(batch, loss_fn)
    rows, _ = probe.reduce(deep=True)

    for name in {r["name"] for r in rows}:
        series = sorted((r["N"], r["PR_Sigma"]) for r in rows if r["name"] == name)
        prs = [pr for _, pr in series]
        assert prs == sorted(prs), f"{name}: PR did not increase with N: {series}"


def test_deep_artifacts_have_expected_shapes(model, cfg, batch):
    probe = GramProbe(model, cfg)
    probe.plan(batch)
    probe.run(batch, loss_fn)
    _, artifacts = probe.reduce(deep=True)

    name = probe.targets[0].name
    assert artifacts[f"{name}|eig_K"].shape == (cfg.top_eig,)
    assert artifacts[f"{name}|eig_M"].shape == (cfg.top_eig,)
    assert artifacts[f"{name}|Xc"].shape[0] == cfg.n_tokens_deep
    assert artifacts[f"{name}|Delta"].shape[0] == cfg.n_tokens_deep
    # Eigenvalues descending, and Gram matrices are PSD.
    eig = artifacts[f"{name}|eig_K"]
    assert (eig[:-1] >= eig[1:] - 1e-4).all()
    assert eig[-1] >= -1e-3


def test_supervised_only_delta_positions_selects_fewer_tokens(model, batch):
    """`delta_positions: supervised` must actually restrict the pool -- half
    of each fixture sequence is masked."""
    common = dict(n_tokens=1000, n_tokens_sweep=(), layer_stride=2, always_layers=(0,))
    all_pos = GramProbe(model, TelemetryConfig(delta_positions="all", **common)).plan(batch)
    sup_pos = GramProbe(model, TelemetryConfig(delta_positions="supervised", **common)).plan(batch)

    assert all_pos["total_valid"] == 2 * 2 * 16          # every non-pad token
    assert sup_pos["total_valid"] == all_pos["total_valid"] // 2


def test_padding_is_excluded_from_the_pool(model):
    """Padded positions carry meaningless activations and must never enter a
    Gram matrix."""
    seq, pad = 16, 6
    ids = torch.randint(3, 128, (2, seq))
    attn = torch.ones(2, seq, dtype=torch.long)
    attn[:, seq - pad:] = 0
    labels = ids.clone()
    labels[attn == 0] = IGNORE_INDEX
    mb = [{"input_ids": ids, "labels": labels, "attention_mask": attn}]

    cfg = TelemetryConfig(n_tokens=1000, n_tokens_sweep=(), layer_stride=2, always_layers=(0,))
    plan = GramProbe(model, cfg).plan(mb)
    assert plan["total_valid"] == 2 * (seq - pad)


def test_probe_survives_a_module_that_never_fires(model, cfg, batch):
    """A missing module must be skipped, not reported with fabricated numbers."""
    probe = GramProbe(model, cfg)
    probe.plan(batch)
    probe.run(batch, loss_fn)
    dropped = probe.targets[0].name
    probe._X.pop(dropped)

    rows, _ = probe.reduce(deep=False)
    assert dropped not in {r["name"] for r in rows}
    assert len(rows) == len(probe.targets) - 1

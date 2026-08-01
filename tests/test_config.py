"""Config layering and validation.

Most of these guard against a run that looks fine and measures the wrong thing:
a misspelled key silently ignored, a batch size that cannot be honoured, a
probe pool too small for the requested token count.
"""

from __future__ import annotations

import pytest

from sftlens.config import CONFIG_ROOT, RecipeConfig, RunConfig, load_config, validate


class TestRecipeArithmetic:
    def test_grad_accum_honours_the_effective_batch(self):
        r = RecipeConfig(effective_batch=32, per_device_batch=2)
        assert r.grad_accum == 16

    def test_indivisible_batch_is_rejected(self):
        """Silently rounding here would change the recipe's batch size, which
        is the one number the whole no-tuning argument rests on."""
        with pytest.raises(ValueError, match="not divisible"):
            _ = RecipeConfig(effective_batch=32, per_device_batch=5).grad_accum

    def test_published_lr_bs_ratios_are_reproduced(self):
        """The ratio SmolTulu identifies as controlling, from Table 2."""
        for lr, bs, ratio in ((3.1e-6, 32, 0.097), (9.0e-5, 8, 11.25), (5.0e-6, 128, 0.039)):
            actual = RecipeConfig(lr=lr, effective_batch=bs).lr_over_bs_e6
            assert actual == pytest.approx(ratio, rel=5e-3)


class TestShippedConfigs:
    """Every checked-in overlay must load and validate."""

    @pytest.mark.parametrize("overlay", [
        ["model/smollm2-1.7b.yaml", "data/tulu3-50k.yaml",
         "recipe/smoltulu-sft-1207.yaml", "telemetry/default.yaml"],
        ["model/smollm2-1.7b.yaml", "data/tulu3-50k.yaml",
         "recipe/smoltulu-sft-1130.yaml", "telemetry/default.yaml"],
        ["model/smollm2-1.7b.yaml", "data/dolly-smoke.yaml",
         "recipe/smoltulu-sft-1207.yaml", "telemetry/smoke.yaml"],
    ])
    def test_overlay_loads(self, overlay):
        cfg = load_config(overlay)
        assert cfg.recipe.grad_accum >= 1

    def test_1207_matches_the_published_row(self):
        cfg = load_config(["model/smollm2-1.7b.yaml", "data/tulu3-50k.yaml",
                           "recipe/smoltulu-sft-1207.yaml", "telemetry/default.yaml"])
        assert cfg.recipe.lr == pytest.approx(3.1e-6)
        assert cfg.recipe.effective_batch == 32
        assert cfg.recipe.epochs == 2.0
        assert cfg.recipe.lr_scheduler == "linear"
        assert cfg.recipe.warmup_ratio == 0.03
        assert cfg.recipe.weight_decay == 0.0
        assert cfg.model.model_id == "HuggingFaceTB/SmolLM2-1.7B"
        assert cfg.data.dataset_id == "allenai/tulu-3-sft-mixture"

    def test_1130_matches_the_published_row(self):
        cfg = load_config(["model/smollm2-1.7b.yaml", "data/tulu3-50k.yaml",
                           "recipe/smoltulu-sft-1130.yaml", "telemetry/default.yaml"])
        assert cfg.recipe.lr == pytest.approx(9.0e-5)
        assert cfg.recipe.effective_batch == 8

    def test_arms_differ_only_in_lr_and_batch(self):
        """The two arms are a controlled learning-rate axis. If anything else
        differs, the comparison is confounded."""
        common = ["model/smollm2-1.7b.yaml", "data/tulu3-50k.yaml", "telemetry/default.yaml"]
        a = load_config([*common, "recipe/smoltulu-sft-1207.yaml"])
        b = load_config([*common, "recipe/smoltulu-sft-1130.yaml"])

        assert a.data == b.data
        assert a.model == b.model
        assert a.telemetry == b.telemetry
        differing = {
            f for f in ("lr", "effective_batch", "epochs", "lr_scheduler", "warmup_ratio",
                        "weight_decay", "max_grad_norm", "optim", "loss_reduction")
            if getattr(a.recipe, f) != getattr(b.recipe, f)
        }
        assert differing == {"lr", "effective_batch"}

    def test_token_cadence_keeps_the_arms_on_a_common_axis(self):
        """At 4x different batch sizes a step-based cadence would put the arms
        on grids that cannot be overlaid."""
        cfg = load_config(["telemetry/default.yaml"])
        assert cfg.telemetry.cadence_unit == "tokens"


class TestOverrides:
    def test_dotted_override_applies_and_is_typed(self):
        cfg = load_config([], ["recipe.lr=1e-5", "telemetry.enabled=false", "seed=7"])
        assert cfg.recipe.lr == pytest.approx(1e-5)
        assert cfg.telemetry.enabled is False
        assert cfg.seed == 7

    def test_later_overlay_wins(self):
        cfg = load_config(["recipe/smoltulu-sft-1207.yaml", "recipe/smoltulu-sft-1130.yaml"])
        assert cfg.recipe.lr == pytest.approx(9.0e-5)

    def test_unknown_key_is_rejected(self):
        """Dropping a misspelled key is how a run ends up not using the setting
        you thought you set."""
        with pytest.raises(ValueError, match="unknown config keys"):
            load_config([], ["recipe.learnign_rate=1e-5"])

    def test_malformed_override_is_rejected(self):
        with pytest.raises(ValueError, match="dotted.key=value"):
            load_config([], ["recipe.lr"])

    def test_yaml_lists_become_tuples(self):
        cfg = load_config([], ["telemetry.n_tokens_sweep=[128, 256]"])
        assert cfg.telemetry.n_tokens_sweep == (128, 256)

    @pytest.mark.parametrize("literal", ["1e-5", "1E-5", "3.1e-6", "-2e3", "+1.5e-8"])
    def test_unpointed_scientific_notation_becomes_a_float(self, literal):
        """YAML 1.1 only reads exponent form as a float when a decimal point is
        present, so `1e-5` parses as the string "1e-5". Unhandled, that reaches
        TrainingArguments as a string learning rate."""
        cfg = load_config([], [f"recipe.lr={literal}"])
        assert isinstance(cfg.recipe.lr, float)
        assert cfg.recipe.lr == pytest.approx(float(literal))

    def test_words_are_not_coerced_to_numbers(self):
        cfg = load_config([], ["recipe.lr_scheduler=linear", "run_name=arm-1130"])
        assert cfg.recipe.lr_scheduler == "linear"
        assert cfg.run_name == "arm-1130"


def test_shipped_recipe_numerics_are_numeric():
    """Guards the same YAML 1.1 gap in the checked-in overlays: a learning rate
    silently read as a string would train at a value nobody chose."""
    for overlay in ("recipe/smoltulu-sft-1207.yaml", "recipe/smoltulu-sft-1130.yaml"):
        cfg = load_config([overlay])
        for f in ("lr", "epochs", "warmup_ratio", "weight_decay", "max_grad_norm",
                  "adam_beta1", "adam_beta2", "adam_epsilon"):
            assert isinstance(getattr(cfg.recipe, f), float), f"{overlay}:{f}"
        assert isinstance(cfg.recipe.effective_batch, int)


class TestValidation:
    def _cfg(self, **telemetry):
        cfg = RunConfig()
        for k, v in telemetry.items():
            setattr(cfg.telemetry, k, v)
        return cfg

    def test_deep_cadence_must_land_on_the_light_grid(self):
        with pytest.raises(ValueError, match="multiple of light_every"):
            validate(self._cfg(light_every=3000, deep_every=10000))

    def test_sweep_cannot_exceed_n_tokens(self):
        with pytest.raises(ValueError, match="exceeds"):
            validate(self._cfg(n_tokens=1024, n_tokens_sweep=(512, 2048)))

    def test_probe_pool_must_supply_n_tokens(self):
        with pytest.raises(ValueError, match="probe pool"):
            validate(self._cfg(probe_seqs=2, probe_max_len=128, n_tokens=8192,
                               n_tokens_sweep=()))

    def test_probe_length_cannot_exceed_training_length(self):
        cfg = RunConfig()
        cfg.telemetry.probe_max_len = 8192
        cfg.data.max_seq_len = 4096
        with pytest.raises(ValueError, match="probe_max_len exceeds"):
            validate(cfg)

    def test_bad_enum_values_are_rejected(self):
        for field, value, match in (
            ("loss_reduction", "median", "sum|mean"),
            ("on_overflow", "wrap", "drop|truncate"),
        ):
            cfg = RunConfig()
            section = cfg.recipe if field == "loss_reduction" else cfg.data
            setattr(section, field, value)
            with pytest.raises(ValueError, match=match):
                validate(cfg)

    def test_disabled_telemetry_skips_probe_validation(self):
        """Turning telemetry off must not be blocked by probe geometry."""
        validate(self._cfg(enabled=False, probe_seqs=1, probe_max_len=8, n_tokens=99999))


def test_run_config_serialises_derived_values(tmp_path):
    cfg = RunConfig()
    cfg.save(tmp_path / "run_config.json")
    payload = cfg.to_dict()
    assert payload["_derived"]["grad_accum"] == cfg.recipe.grad_accum
    assert payload["_derived"]["lr_over_bs_e6"] == pytest.approx(cfg.recipe.lr_over_bs_e6)
    assert (tmp_path / "run_config.json").exists()


def test_output_dir_interpolates_run_name():
    cfg = RunConfig(run_name="arm-a", output_dir="runs/${run_name}")
    assert cfg.resolved_output_dir().name == "arm-a"


def test_config_root_points_at_the_shipped_configs():
    assert (CONFIG_ROOT / "base.yaml").exists()

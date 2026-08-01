"""Training entrypoint.

    sftlens-train -c model/smollm2-1.7b.yaml -c data/tulu3-50k.yaml \
                  -c recipe/smoltulu-sft-1207.yaml -c telemetry/default.yaml

Resumes automatically from the latest checkpoint in the output directory.
"""

from __future__ import annotations

import argparse
import json
import math
import weakref
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from ..config import RunConfig, load_config
from ..data import ChatMLTemplate, PadCollator, build_datasets, describe_masking
from ..telemetry.callback import attach_telemetry
from ..utils.env import (
    check_capacity,
    describe_environment,
    estimate_logits_gb,
    estimate_probe_accum_gb,
    estimate_training_state_gb,
    seed_everything,
)
from .trainer import SFTTrainer, TrainLogCallback


def build_tokenizer(cfg: RunConfig):
    tok = AutoTokenizer.from_pretrained(cfg.model.model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def build_model(cfg: RunConfig, tokenizer):
    dtype = getattr(torch, cfg.model.param_dtype)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.model_id,
        torch_dtype=dtype,
        attn_implementation=cfg.model.attn_implementation,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    return model


def build_training_args(cfg: RunConfig, out_dir: Path) -> TrainingArguments:
    r = cfg.recipe
    return TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=r.epochs,
        max_steps=r.max_steps,
        per_device_train_batch_size=r.per_device_batch,
        per_device_eval_batch_size=r.per_device_batch,
        gradient_accumulation_steps=r.grad_accum,

        learning_rate=r.lr,
        lr_scheduler_type=r.lr_scheduler,
        warmup_ratio=r.warmup_ratio,
        weight_decay=r.weight_decay,
        max_grad_norm=r.max_grad_norm,
        optim=r.optim,
        adam_beta1=r.adam_beta1,
        adam_beta2=r.adam_beta2,
        adam_epsilon=r.adam_epsilon,

        bf16=cfg.model.bf16,
        gradient_checkpointing=cfg.model.gradient_checkpointing,
        # Non-reentrant checkpointing is required, not preferred: the probe's
        # full backward hooks do not fire correctly under the reentrant path.
        gradient_checkpointing_kwargs=(
            {"use_reentrant": False} if cfg.model.gradient_checkpointing else None
        ),

        logging_steps=cfg.logging_steps,
        eval_strategy=cfg.eval_strategy,
        eval_steps=cfg.eval_steps,
        save_strategy=cfg.save_strategy,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=False,

        seed=cfg.seed,
        data_seed=cfg.seed,
        dataloader_num_workers=cfg.dataloader_num_workers,
        group_by_length=cfg.group_by_length,
        use_cpu=cfg.use_cpu,
        length_column_name="length",
        # Our collator reads input_ids/labels and ignores the rest; `source` has
        # to survive so the probe batch can be stratified by it.
        remove_unused_columns=False,
        report_to=cfg.report_to,
    )


def save_final(trainer, tokenizer, template, cfg: RunConfig, out_dir: Path) -> None:
    """Write the deployable model.

    Saved in bf16 from a fp32 copy rather than by casting the live model: an
    in-place cast destroys the fp32 master weights, so the run could not be
    resumed or re-evaluated afterwards.
    """
    final = out_dir / "final"
    final.mkdir(parents=True, exist_ok=True)

    # Cast a COPY of the state dict. `model.to(bfloat16)` would work but
    # destroys the fp32 master weights in place, leaving the live model unable
    # to resume or to be re-evaluated at training precision.
    state = {k: v.detach().to(torch.bfloat16) for k, v in trainer.model.state_dict().items()}
    trainer.model.save_pretrained(final, state_dict=state, safe_serialization=True)
    tokenizer.save_pretrained(final)
    del state

    # The base model's eos is <|endoftext|>, but training taught it to stop on
    # <|im_end|>. Without this the model looks broken and rambles past the turn.
    generation = {
        "eos_token_id": sorted({template.im_end_id, tokenizer.eos_token_id}),
        "pad_token_id": tokenizer.pad_token_id,
    }
    (final / "generation_config.json").write_text(json.dumps(generation, indent=2))
    cfg.save(final / "run_config.json")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="sftlens-train")
    parser.add_argument("-c", "--config", action="append", default=[],
                        help="YAML overlay, repeatable; later files win")
    parser.add_argument("-s", "--set", action="append", default=[], dest="overrides",
                        help="dotted override, e.g. recipe.lr=3.1e-6")
    parser.add_argument("--dry-run", action="store_true",
                        help="build data and model, report sizing, then stop")
    parser.add_argument("--no-save-final", action="store_true",
                        help="skip writing the final model (smoke runs do not need it)")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, args.overrides)
    out_dir = cfg.resolved_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg.seed)

    env = describe_environment()
    print(f"[env] {json.dumps(env, indent=2)}")
    print(f"[recipe] {cfg.recipe.provenance}")
    print(
        f"[recipe] lr={cfg.recipe.lr:g} effective_batch={cfg.recipe.effective_batch} "
        f"(= {cfg.recipe.per_device_batch} x {cfg.recipe.grad_accum}) "
        f"LR/BS={cfg.recipe.lr_over_bs_e6:.3f}e-6"
    )

    tokenizer = build_tokenizer(cfg)
    template = ChatMLTemplate(
        tokenizer=tokenizer,
        system_prompt=cfg.data.system_prompt,
        im_start=cfg.model.im_start,
        im_end=cfg.model.im_end,
    )
    train_ds, eval_ds = build_datasets(template, cfg.data, cfg.seed)

    # Label-masking bugs are silent: the loss curve looks healthy while the
    # model learns to parrot prompts. Always eyeball this before spending GPU.
    print(describe_masking(template, train_ds[0]))

    model = build_model(cfg, tokenizer)
    collator = PadCollator(tokenizer.pad_token_id)

    n_params = sum(p.numel() for p in model.parameters())
    mem = estimate_training_state_gb(n_params, cfg.model.param_dtype)
    print(f"[env] {n_params / 1e9:.2f}B params; {json.dumps(mem)}")

    training_args = build_training_args(cfg, out_dir)
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        loss_reduction=cfg.recipe.loss_reduction,
    )
    trainer.add_callback(TrainLogCallback(out_dir / "train_log.jsonl", weakref.ref(trainer)))

    # HF's Trainer places the model on `args.device` inside __init__, so this
    # already reflects final placement even before training starts. Stated
    # explicitly rather than left to be inferred from throughput after the
    # fact: a CPU fallback here is not a slow run, it is a run that silently
    # never finishes at production scale, and the failure mode (Python-level
    # device mismatch) has no other visible symptom until someone notices the
    # wall clock looks wrong.
    model_device = next(model.parameters()).device
    print(f"[env] model is on device: {model_device}")
    if torch.cuda.is_available() and not cfg.use_cpu and model_device.type != "cuda":
        raise RuntimeError(
            f"CUDA is available and use_cpu=False, but the model ended up on "
            f"{model_device} instead of a CUDA device. Full-parameter "
            f"training of a model this size on CPU would take orders of "
            f"magnitude longer than on GPU rather than merely running slowly, "
            f"so this is treated as a hard stop rather than a warning."
        )

    telemetry = attach_telemetry(trainer, cfg, eval_ds, collator)
    probe_gb = (
        estimate_probe_accum_gb(telemetry.probe.targets, cfg.telemetry.n_tokens)
        if telemetry is not None else 0.0
    )
    # The longest batch sets the logits peak, so budget against the longest
    # sequence that survived the length filter, not the mean.
    longest = max(train_ds["length"])
    check_capacity(
        env, mem["total_static_gb"], probe_gb,
        estimate_logits_gb(cfg.recipe.per_device_batch, longest, model.config.vocab_size),
        f"(batch {cfg.recipe.per_device_batch} x longest seq {longest})",
    )

    cfg.save(out_dir / "run_config.json")
    (out_dir / "environment.json").write_text(json.dumps(env, indent=2))

    if args.dry_run:
        print("[run] dry run complete; stopping before training")
        return

    resume = get_last_checkpoint(str(out_dir))
    if resume:
        print(f"[run] resuming from {resume}")
    trainer.train(resume_from_checkpoint=resume)

    metrics = trainer.evaluate()
    if "eval_loss" in metrics:
        metrics["perplexity"] = math.exp(min(metrics["eval_loss"], 20.0))
    print(f"[final] {metrics}")
    (out_dir / "final_metrics.json").write_text(json.dumps(metrics, indent=2))

    if args.no_save_final:
        print("[run] skipping final model export (--no-save-final)")
    else:
        save_final(trainer, tokenizer, template, cfg, out_dir)
    print(f"[run] done -> {out_dir}")


if __name__ == "__main__":
    main()

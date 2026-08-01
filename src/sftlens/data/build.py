"""Dataset assembly: raw rows -> normalized conversations -> tokenized splits.

Every supported dataset is normalized to a single `messages` schema first, so
there is exactly one tokenization and masking path. A second encoder per dataset
would be a second place for the label mask to be wrong.
"""

from __future__ import annotations

from datasets import Dataset, load_dataset

from ..config import DataConfig
from .chatml import ChatMLTemplate
from .mixture import format_composition, stratified_indices

# Datasets carrying no mixture label are treated as a single stratum, so the
# stratification and reporting code has no special case for them.
UNLABELED = "__single__"


# ---------------------------------------------------------------------------
# Normalizers: dataset-specific rows -> {"messages": [...], "source": str}
# ---------------------------------------------------------------------------
def _normalize_tulu3(row: dict) -> dict:
    return {"messages": row["messages"], "source": row.get("source", UNLABELED)}


def _normalize_dolly(row: dict) -> dict:
    """dolly-15k is flat instruction/context/response, single turn.

    Context is folded into the user turn rather than a system turn: it is
    reference material for the specific question, not a persistent instruction.
    """
    user = row["instruction"].strip()
    context = (row.get("context") or "").strip()
    if context:
        user = f"{user}\n\n{context}"
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": row["response"].strip()},
        ],
        "source": row.get("category", UNLABELED),
    }


NORMALIZERS = {"tulu3": _normalize_tulu3, "dolly": _normalize_dolly}


# ---------------------------------------------------------------------------
def load_conversations(cfg: DataConfig, seed: int) -> Dataset:
    """Load, normalize, and proportionally subsample to `cfg.subset_size`."""
    if cfg.loader not in NORMALIZERS:
        raise ValueError(f"unknown data.loader {cfg.loader!r}; have {sorted(NORMALIZERS)}")

    raw = load_dataset(cfg.dataset_id, split=cfg.split)
    ds = raw.map(
        NORMALIZERS[cfg.loader],
        remove_columns=raw.column_names,
        num_proc=cfg.num_proc,
        desc="normalizing",
    )

    # Subsample BEFORE tokenizing: tokenizing 939k conversations to then discard
    # 95% of them wastes most of the preprocessing budget.
    if cfg.subset_size is not None and cfg.subset_size < len(ds):
        labels = ds[cfg.stratify_column] if cfg.stratify_column else [UNLABELED] * len(ds)
        idx = stratified_indices(labels, cfg.subset_size, seed)
        ds = ds.select(idx)

    print(format_composition(ds["source"], "subset composition"))
    return ds


def encode(ds: Dataset, template: ChatMLTemplate, cfg: DataConfig) -> Dataset:
    """Tokenize with loss masking, then enforce the length budget."""

    def _encode(row: dict) -> dict:
        out = template.encode(row["messages"])
        out["source"] = row["source"]
        return out

    encoded = ds.map(
        _encode,
        remove_columns=ds.column_names,
        num_proc=cfg.num_proc,
        desc="tokenizing",
    )

    before = len(encoded)
    if cfg.on_overflow == "drop":
        encoded = encoded.filter(
            lambda r: r["length"] <= cfg.max_seq_len,
            num_proc=cfg.num_proc,
            desc="length filter",
        )
        dropped = before - len(encoded)
        print(
            f"[data] dropped {dropped}/{before} ({100 * dropped / max(before, 1):.1f}%) "
            f"conversations over {cfg.max_seq_len} tokens"
        )
        if dropped / max(before, 1) > 0.15:
            # Heavy dropping silently reshapes the mixture: long-form sources go
            # first, so the composition no longer matches what the LR was picked
            # against.
            print(
                f"[data] WARNING: >15% dropped at max_seq_len={cfg.max_seq_len}. "
                "The surviving subset is biased toward short sources; either "
                "raise max_seq_len or report the post-filter composition."
            )
            print(format_composition(encoded["source"], "post-filter composition"))
    else:
        def _truncate(r: dict) -> dict:
            n = cfg.max_seq_len
            return {
                "input_ids": r["input_ids"][:n],
                "labels": r["labels"][:n],
                "length": min(r["length"], n),
            }

        encoded = encoded.map(_truncate, num_proc=cfg.num_proc, desc="truncating")

    return encoded


def build_datasets(
    template: ChatMLTemplate, cfg: DataConfig, seed: int
) -> tuple[Dataset, Dataset]:
    """Return (train, eval). The probe batch is drawn from eval, so it is
    held out from the optimizer for the whole run."""
    conversations = load_conversations(cfg, seed)
    encoded = encode(conversations, template, cfg)

    if cfg.eval_size >= len(encoded):
        raise ValueError(
            f"eval_size={cfg.eval_size} but only {len(encoded)} examples survived"
        )
    split = encoded.train_test_split(test_size=cfg.eval_size, seed=seed)
    train, evaluation = split["train"], split["test"]

    print(f"[data] train={len(train)}  eval={len(evaluation)}")
    print(
        f"[data] tokens: train={sum(train['length']):,}  "
        f"supervised={sum(train['n_supervised']):,}"
    )
    return train, evaluation

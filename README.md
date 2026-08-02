# SFTLens

SFTLens watches what happens inside a language model's layers while it is
being fine-tuned. Instead of only tracking the loss curve, it measures how
the internal structure of each layer changes over the course of training —
a more direct look at what fine-tuning is actually doing to the model.

## Why

Most fine-tuning pipelines report a loss curve and a benchmark score and
treat everything in between as a black box. SFTLens adds lightweight
instrumentation that periodically samples each layer's activations and
gradients during training, without slowing the run down or changing its
outcome, and turns the samples into a small set of interpretable statistics.

The training recipe itself is copied from a published paper and applied
without any tuning, so anything interesting in the measurements can be
attributed to the training process itself rather than to hyperparameter
choices made along the way.

## Results

The full pipeline was run end to end on an H100 GPU. Complete results,
including data tables and figures, are in
[`analysis/dolly_full_dryrun/FINDINGS.md`](analysis/dolly_full_dryrun/FINDINGS.md). The tables are also provided here [`analysis/dolly_full_dryrun/tables`](analysis/dolly_full_dryrun/tables)

Highlights:

- How closely a layer's gradient aligns with its input structure starts high
  and decays toward a "no special alignment" baseline as training proceeds —
  a clear, repeatable trend that holds up under several statistical checks.
- One early result turned out to rest on a flawed comparison between two
  measurements taken on different scales. The comparison was corrected,
  checked against an independent calculation, and the analysis was re-run.
  The original result did not hold up under the corrected version, and this
  is reported as a retraction in the findings rather than quietly removed.

## How it works

At intervals during training, on a small fixed batch of held-out data,
SFTLens records:

- how spread out each layer's inputs are across different directions
- how much the gradient lines up with that input structure
- the same kind of statistics for the output side
- a comparison against what these numbers would look like if there were no
  real structure at all, as a sanity check

Full statistical definitions, and a discussion of why naive versions of the
main measurement are misleading at small sample sizes, are in
[`docs/methodology.md`](docs/methodology.md).

The training recipe (model, dataset, hyperparameters) is described in the
same document.

## Installation and usage

See [`docs/runpod.md`](docs/runpod.md) for cloud GPU setup, sizing, and cost.

```bash
pip install -e ".[dev]"

# environment check, no downloads
python scripts/preflight.py

# end-to-end pipeline check, a few minutes
bash scripts/smoke.sh

# full training run
sftlens-train \
  -c model/smollm2-1.7b.yaml \
  -c data/tulu3-50k.yaml \
  -c recipe/smoltulu-sft-1207.yaml \
  -c telemetry/default.yaml
```

`--dry-run` builds the data and model, reports memory sizing, and exits
before training. `-s dotted.key=value` overrides any config field. A run
resumes automatically from the latest checkpoint in its output directory.

## Project structure

```
src/sftlens/       the package: config, data loading, telemetry, training
configs/           YAML configs for model / data / recipe / telemetry
tests/             129 tests
analysis/          experiment write-ups; start at dolly_full_dryrun/FINDINGS.md
docs/              methodology and cloud setup guides
scripts/           environment checks, smoke tests, cloud sync utilities
```

## License

[Apache-2.0](LICENSE)

# Full-training validation of CLAS on 20 reassembled architectures

This evidence package links architecture configurations, completed 100-epoch
ImageNet-1K training logs, and initial-state proxy scores through the anonymous
architecture IDs **A01–A20**. These IDs are stable throughout this package;
training runs and scoring seeds do not count as additional architectures.

## Results

See [the complete correlation table](table.md) and
[full-precision correlations](correlations.csv). CLAS obtains Spearman
**0.807 ± 0.020**, compared with **0.707 ± 0.009** when only the square-root
transformation is removed. The table includes all eight evaluated proxies
and the CLAS ablation. Each correlation is computed across the same 20
architectures separately for scoring seeds 11, 17, and 23. We then report the
mean and sample standard deviation of those three correlations. This is not
the correlation obtained by averaging proxy scores before computing ranks,
and the SD does not describe variation across training seeds.

## Files and correspondence

| File | Contents |
|---|---|
| [architectures.csv](architectures.csv) | One row per architecture: anonymous ID, final Top-1, aggregation, and configuration |
| [configs/](configs/) | Archived architecture build/scoring configurations, with comments and identifying paths removed |
| [training_runs.csv](training_runs.csv) | Training run IDs, recorded training seeds, epoch-100 Top-1, and log segment paths |
| [training/](training/) | 25 JSONL log segments from 24 training runs, preserving all original train/validation metric records |
| [proxy/scores.csv](proxy/scores.csv) | 540 scores: 20 architectures × 3 scoring seeds × 9 proxy/ablation entries |
| [proxy/audit_records.jsonl](proxy/audit_records.jsonl) | 60 underlying scoring records, including implementation variants and initial/final state hashes |
| [proxy/score_definitions.json](proxy/score_definitions.json) | Exact audit field used for each reported proxy |
| [proxy/protocol.json](proxy/protocol.json) | Scoring environment, calibration tensor/label hashes, seeds, and protocol |
| [proxy/calibration_images.json](proxy/calibration_images.json) | Dataset-relative calibration image names and original file hashes |
| [proxy/implementation/](proxy/implementation/) | Archived scoring kernels and public reference implementations for inspection |
| [provenance.json](provenance.json) | Source/release hashes and anonymization record, without identifying source paths |
| [checksums.json](checksums.json) | SHA-256 hashes of the released files |

For example, `A01` in `architectures.csv` links to `configs/A01.py`, its
training run in `training_runs.csv`, and its rows in `proxy/scores.csv`.
The underlying audit records also use `architecture_id: A01`.

## Recompute and verify

Run from the repository root with Python 3.8 or newer; no third-party packages,
GPU, dataset, or checkpoints are needed to verify the released measurements:

```bash
python3 evidence/clas_100e_20/recompute.py
```

The script verifies every architecture's final target against its epoch-100
validation log, checks all 540 reported scores against the underlying audit
fields, confirms matching initial/final model state hashes, checks file
checksums, and regenerates `correlations.csv` and `table.md`.

This command recomputes the statistics from recorded experiments. The archived
scoring kernels are provided for inspection; they are not a self-contained
one-command GPU scoring distribution. Re-running training or scoring requires
the appropriate legacy environment, ImageNet data, and pretrained sources.
Archived configurations retain their historical model registry names.

## Training and scoring protocol

All 20 architectures completed their prescribed 100-epoch ImageNet-1K schedule.
The final target is epoch-100 validation Top-1, rather than best-epoch accuracy.
A12 and A13 each use the mean of training seeds 0/11/23; each remains one
architecture. A12's first training run has two log segments from the same run.
The other architectures use one completed training run each. Missing training
seeds are explicitly recorded as `not_recorded`, never inferred as seed 0.

All proxies use the same initial models and 32-image calibration panel, with
fresh scoring initializations 11/17/23. Source blocks use their pretrained
weights; no final trained checkpoint is loaded and no optimizer step is taken.
CLAS and its no-square-root ablation share the monitored modules and distinct
activation-pattern counts. The ablation removes only the per-module square
root. It is not the separate Raw SWAP score.

The main table uses the shared evaluation-mode comparison: reference-aligned
GradNorm, SNIP, Fisher, NASWOT, ZiCo, Jacov, and SynFlow definitions are specified
by the exact paths in `proxy/score_definitions.json`. SynFlow is stored on a
logarithmic scale, preserving its rank. Alternative contexts and diagnostic
variants remain in the audit records and are not substituted according to
which result is highest. Calibration image pixels and model weights are not
included; image names/hashes, labels, and preprocessing are recorded instead.

## Cohort and interpretation

The fixed cohort contains eight independently sampled candidates (A01–A08)
and twelve searched or selected candidates (A09–A20), spanning two resource
budgets. It is not 20 independent random draws. The additional seven eligible
architectures were fixed before their proxy scoring and all seven are retained.
The correlation supports ranking utility on this cohort; it does not establish
size-independent superiority, statistical significance of the square-root
improvement, or a guarantee of selecting the best individual architecture.

## Anonymization

Only identifiers, filesystem paths, non-metric log metadata, and cosmetic
source-code whitespace were changed
or removed. All train/validation metric records and proxy values are preserved.
The log files are anonymized metric-record exports, not byte-identical copies
of the original server logs. Their release hashes therefore differ from source
hashes. `provenance.json` records both. The private correspondence between
original experiment names and anonymous IDs is not part of this release.

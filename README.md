# UGVU: Uncertainty-Guided Generative Vision Understanding

UGVU is a Python research prototype for reliability diagnosis of dense predictions from black-box vision-language and generative vision APIs. It repeatedly queries one or more endpoints, decodes outputs into semantic masks, estimates consensus uncertainty, fuses predictions, and reports calibration and failure diagnostics.

This GitHub repository is the cleaned code release. Manuscript drafts, paper figures, local datasets, generated outputs, and one-off workspace artifacts are intentionally not tracked.

## Core Features

- End-to-end dense-prediction pipeline for semantic segmentation.
- Black-box API wrappers for Qwen/DashScope, Doubao/Volcengine, Gemini-style REST endpoints, Flux, and RightCode-compatible image endpoints.
- Output decoders for generated images, JSON/grid-style responses, RLE masks, and class-index masks.
- Pixel-wise Consensus Uncertainty (PCU) from repeated decoded outputs.
- Majority voting, Uncertainty-Guided Consensus Fusion (UGCF), and Cross-Model Consensus Fusion (CMCF).
- Calibration and reliability metrics including mIoU, ECE, Spearman correlation, AUROC, and failure tags.
- Unit tests and small synthetic/API sanity-check entry points.

## Repository Layout

```text
ugvu/                              Core package
  benchmark/                       Robustness benchmark runner
  calibration/                     ECE, correlation, reliability plots
  consensus/                       Majority vote, UGCF, CMCF
  configs/                         YAML config dataclasses and loader
  datasets/                        Dataset adapters and label metadata
  decoders/                        API output to dense-mask decoders
  generators/                      API generator wrappers and mock generator
  metrics/                         Segmentation and dense-prediction metrics
  prompts/                         Prompt templates and variants
  robustness/                      Prompt/model/generation robustness scores
  uncertainty/                     Entropy, variance, and uncertainty maps
  visualization/                   Comparison and uncertainty visualizations

configs/                           Example YAML configs
tests/                             Unit and diagnostic tests
experiments/                       Structured experiment entry scripts
run_api_sanity.py                  Small synthetic-data API sanity check
run_requested_experiments.py       Candidate-constrained diagnostic runner
run_*_experiment.py                Dataset/API experiment runners
summarize_diagnostic_results.py    JSON result summarizer
```

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

After editable installation, both forms are available:

```bash
ugvu --help
python -m ugvu --help
```

## API Keys

Create a local `.env` file or export the environment variables needed by the endpoints you run. `.env` is ignored by Git.

```bash
DASHSCOPE_API_KEY=your_dashscope_key
DOUBAO_API_KEY=your_doubao_key
IMAGE_API_KEY=your_rightcode_key
GOOG_API_KEY=your_google_key
FLUX_API_KEY=your_flux_key
```

Only the keys for selected models are required.

## Basic Usage

Run the full pipeline:

```bash
python -m ugvu run -c configs/default.yaml -t semantic_segmentation -o outputs/run1
```

Evaluate saved predictions:

```bash
python -m ugvu evaluate -c configs/default.yaml -t semantic_segmentation -o outputs/run1
```

Run calibration analysis:

```bash
python -m ugvu calibrate -c configs/default.yaml -t semantic_segmentation -o outputs/run1
```

Run robustness analysis:

```bash
python -m ugvu robustness -c configs/default.yaml -t semantic_segmentation -p 50 -o outputs/run1
```

Generated predictions, uncertainty maps, cached endpoint responses, metrics, and figures are written under `outputs/` by default and are not tracked.

## API Sanity Check

`run_api_sanity.py` uses the synthetic Cityscapes fixture under `data/synthetic_cityscapes` when available and checks whether selected remote generators can complete one small pipeline run.

```bash
python run_api_sanity.py --models qwen-vl,doubao
python run_api_sanity.py --models gpt-image-2,nano-banana-2
```

API runs require network access and valid keys.

## Candidate-Constrained Diagnostic Runs

`run_requested_experiments.py` can run small candidate-constrained diagnostics on Cityscapes or ADE20K. Update local dataset roots in the script before running on another machine.

```bash
python run_requested_experiments.py \
  --dataset cityscapes \
  --models doubao,qwen-vl,gpt-image-2,nano-banana-2 \
  --k-values 1,3 \
  --max-samples 20 \
  --output-dir outputs/cityscapes_candidate
```

## Testing

Run the local tests:

```bash
python -m pytest tests
```

The tests in `tests/` use synthetic or mock data. Remote API checks require valid credentials and network access.

## Tracked vs. Local-Only Files

Tracked in GitHub:

- Core package: `ugvu/`
- Configs: `configs/`
- Tests: `tests/`
- Structured experiment scripts: `experiments/`
- Selected root runners: `run_*_experiment.py`, `run_api_sanity.py`, `run_requested_experiments.py`, `summarize_diagnostic_results.py`
- Packaging and dependency files: `pyproject.toml`, `setup.py`, `requirements.txt`

Not tracked:

- Manuscript drafts and paper source
- Paper figures and generated plots
- Local datasets and endpoint outputs
- `.env` and credentials
- Zip archives, cache folders, and one-off workspace scripts

## License

The package metadata declares MIT license terms. Add a standalone `LICENSE` file if a public release requires one.

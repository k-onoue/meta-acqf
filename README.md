# meta-acqf

## Setup

```bash
# Create virtual environment with Python 3.12
uv venv .venv-meta --python 3.12

# Activate environment
source .venv-meta/bin/activate

# Install package in editable mode
uv pip install -e .
```

## Setup Datasets

Results are written under `results/${task}/${model}/${acquisition}/...` by Hydra.

```bash
bash scripts/setup_datasets.sh
```

This will:
1. Download raw datasets to `data/raw/`
2. Run preprocessing to generate oracle embeddings and BoW features
3. Save processed `.pt` files to `data/`

## Train

Hydra config defaults use Reuters + MI acquisition. Swap tasks/acquisitions via CLI overrides.

Paper-aligned defaults:
- Candidate pool: 500 docs per task (1 target + 500 candidates sampled)
- Initial pool: 1 observation
- Query budget T: 10 steps
- MI exploration nu: 14.50865
- Evaluation: average_cumulative_gap, seeds=10, test_tasks=50, val_tasks=20

```bash
# Standard run
python -m meta.main

# Debug-fast run
python -m meta.main train=debug

# Switch dataset/acquisition
python -m meta.main task=neurips acquisition=ucb
```

Results are written under `results/${task}/${model}/${acquisition}/...` by Hydra.

### Batch experiments (paper setup)

Run 10 seeds for each dataset with MI:

```bash
bash scripts/run_experiments.sh
```

### Baseline (no RL) runs

Greedy acquisition baselines for RBF GP and DKL GP (no policy gradient):

```bash
bash scripts/run_baselines.sh
```




https://aistudio.google.com/prompts/1LGSXAze6gNVeyEbifFhDfsBVTsKp9DyS# meta-acqf

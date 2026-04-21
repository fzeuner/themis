# themis
Code for analyzing themis data

## Installation and setup

- **Quick setup (if `uv` is already installed)**:
  - `git clone <repo-url>`
  - `cd themis`
  - `uv sync`
  - This is sufficient to create/use `.venv`, install dependencies from `uv.lock`, and make `themis` imports callable.

- **Create/activate uv-managed `.venv` for THEMIS**:
  - `uv python install 3.14`
  - `uv venv --python 3.14`
  - `source .venv/bin/activate`

- **Sync/install dependencies from `pyproject.toml` and `uv.lock`**:
  - `uv sync`

- **Editable install behavior**:
  - This repo uses `pyproject.toml` with `src/` as package root.
  - `uv sync` installs the project and dependencies into `.venv`, so imports like `from themis...` work without `sys.path` hacks.

- **Atlas-fit stays on separate conda env (unchanged)**:
  - `conda create -n atlas-fit python=3.10 numpy=1.26`
  - `conda activate atlas-fit`
  - `git clone https://gitlab.gwdg.de/hoelken/atlas-fit.git`
  - `pip install -e atlas-fit`

- **Notes**:
  - `atlas-fit` is intentionally **not** installed in the THEMIS `.venv`; calls to atlas-fit scripts run in the `atlas-fit` conda env.



## Configuration-driven datasets (2025)

The module `src/themis/datasets/themis_datasets_2025.py` now supports loading dataset and data-reduction parameters from an external TOML configuration file. This lets you keep multiple dataset-specific configs without changing code.
- `auto_discover_files=True` selects the most relevant file per reduction level using the naming pattern (camera, data type, sequence), preferring `_fx` files when present.

### Create or copy a config

Use the example at `configs/sample_dataset_sr_2025-07-07.toml` as a template. Copy it and edit values for your run:

```
cp configs/sample_dataset_sr_2025-07-07.toml configs/my_run.toml
vim configs/my_run.toml
```

Key sections in the TOML file:

- `[dataset]` — line key, date, sequences, polarization states
- `[paths]` — base directory for rdata/pdata/inversion; optional overrides for figures and inversion
- `[params]` — reduction parameters such as `slit_width`

Example snippet:

```toml
[dataset]
line = "sr"
date = "2025-07-07"
sequence = 26
flat_sequence = 25
dark_sequence = 1
states = ["pQ", "mQ", "pU", "mU", "pV", "mV"]

[paths]
base = "/home/<user>/data/themis"
# figures = "/home/<user>/figures/themis/"     # optional
# inversion = "/home/<user>/data/themis/inversion"  # optional

[params]
slit_width = 0.33
```

### Load a config in Python

Use `get_config(config_path=...)` to populate the dataset variables from your TOML file. If you omit `config_path`, the module falls back to the built-in defaults.

```python
from themis.datasets import themis_datasets_2025 as ds

cfg = ds.get_config(
    config_path="configs/my_run.toml",   # or None to use defaults
    auto_discover_files=True,             # scan directories to pre-fill matching files per level
    auto_create_dirs=False,               # create rdata/pdata/figures/inversion if missing
)

print(cfg.directories)
print(cfg.dataset)
```



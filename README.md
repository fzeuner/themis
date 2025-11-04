# themis
Code for analyzing themis data

## Installation and setup

- **Create/activate conda env** (Python 3.12 recommended):
  - `conda create -n themis python=3.12`
  - `conda activate themis`
- **Install dependencies**:
  - `conda env update -n themis -f environment.yml`
- **Editable install of this package** (so imports like `from themis...` work everywhere and scripts don’t need sys.path hacks):
  - `pip install -e .`
  - This uses `pyproject.toml` with `src/` as the package root.

Get https://github.com/fzeuner/spectator, then:
  pip install -e spectator

Get https://gitlab.gwdg.de/hoelken/atlas-fit/-/tree/main?ref_type=heads, then:
  git clone https://gitlab.gwdg.de/hoelken/atlas-fit.git
  pip install -e atlas-fit

Get https://gitlab.gwdg.de/hoelken/spectroflat:
  pip install spectroflat
  --> Then, remove all pycache files in the spectroflat directory + remove replace np.infty with np.inf in the spectroflat source code smile_fit.py


On a fresh machine, repeat the above steps to get started.

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

### Notes

- Python 3.12 environment is provided; the loader uses the stdlib `tomllib`.
- File types are fixed as `scan`, `dark`, `flat` to match existing code.
- `get_config()` resolves relative config paths automatically. It tries: as given (CWD), relative to project root, and under `project_root/configs/`.
- You can also pass the config path via an env var `THEMIS_CONFIG` (if your script supports it) or via a CLI flag (if implemented).


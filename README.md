# themis
Code for reducing and analyzing themis data

## Installation and setup

- **Quick setup (if `uv` is already installed)**:
  - `git clone https://github.com/fzeuner/themis`
  - `cd themis`
  - `uv sync`


- **Atlas-fit has to be installed in a separate conda env **:
  - `conda create -n atlas-fit python=3.10 numpy=1.26`
  - `conda activate atlas-fit`
  - `git clone https://gitlab.gwdg.de/hoelken/atlas-fit.git`
  - `pip install atlas-fit`


## Configuration-driven datasets (2025)

Use a TOML config file (for example `configs/sample_dataset_sr_2025-07-07.toml`) to define dataset and reduction parameters.

### Load config in Python

```python
from themis.datasets import themis_datasets_2025 as ds

cfg = ds.get_config(
    config_path="configs/my_run.toml",
    auto_discover_files=True,
)

print(cfg.directories)
print(cfg.dataset)
```
### Important information
This is not meant to be exhaustive in explanation how to use the code. Contact me if you want to use the code for your own data!
The dataset from 2024 is not tested with the new code!
This repo is making use of spectroflat (https://gitlab.gwdg.de/hoelken/spectroflat) and atlas-fit (https://gitlab.gwdg.de/hoelken/atlas-fit)
(there might be one place where atlas-fit logic goes wrong, but I still need to check)


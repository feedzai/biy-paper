# Dataset

[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)

## Development

Install [Oxipng](https://github.com/oxipng/oxipng) (if necessary):

```bash
brew install oxipng
```

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) (if necessary):

```bash
curl -LsSf https://astral.sh/uv/0.7.19/install.sh | sh
```

```bash
uv python install
```

```bash
uv venv
```

```bash
source .venv/bin/activate
```

```bash
uv pip install -r requirements.txt
```

```bash
playwright install
```

```bash
ruff check
```

```bash
mypy
```

```bash
ruff check --fix
```

```bash
ruff format
```

Generate the dataset from scratch:

```bash
python generate_datasets.py && \
python generate_vl.py && \
python validate_metadata.py && \
python process_images.py
```

Save the dataset in Parquet for the benchmark:

```bash
python generate_local_dataset.py
```

Preview:

```bash
npx hyperparam output/dataset.parquet
```

```bash
deactivate
```

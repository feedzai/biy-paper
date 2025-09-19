# Benchmark

[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)

## Development

```bash
cp .env.example .env
```

```bash
cp ../dataset/output/dataset.parquet input/dataset.parquet
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
mypy
```

```bash
ruff check --fix
```

```bash
ruff format
```

```bash
deactivate
```

### Run the benchmark

#### OpenAI

```bash
python prepare_open_ai_batches.py
```

```bash
python run_open_ai_batches.py
```

```bash
python check_open_ai_batches.py
```

```bash
python download_open_ai_results.py
```

#### Google

```bash
python prepare_google_batches.py
```

```bash
python upload_google_batches.py
```

```bash
python run_google_batches.py
```

```bash
python download_google_results.py
```

### Evaluate the benchmark

```bash
python generate_results.py
```

```bash
python evaluate_counts.py
```

```bash
python evaluate_consistency.py
```

```bash
python evaluate_cluster_bboxes.py
```

```bash
python evaluate_points.py
```

```bash
python evaluate_chart_designs.py
```

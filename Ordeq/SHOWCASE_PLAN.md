# Ordeq Showcase – Full Context & Plan

## 🎯 Talk Goal
Show everything Ordeq can do through **two practical pipelines**:
1. **Classical ML Pipeline** – preprocessing + training + evaluation (scikit-learn)
2. **LLM Pipeline** – text enrichment / classification with a HuggingFace model

Plus a quick `git bisect` demo at the end.

---

## 📚 Ordeq Core Concepts (to demonstrate)

| Concept | What it is | How we'll show it |
|---|---|---|
| **IO** (`IO`, `Input`, `Output`) | Decouples data loading/saving from logic. IO knows *how* to load/save but holds no data. | `PandasCSV`, `HuggingfaceDataset`, `Joblib`, `NumpyBinary`, `MatplotlibFigure`, `PandasParquet` |
| **Node** (`@node`) | A decorated function with declared `inputs` and `outputs`. Pure transformation. | Every pipeline step is a node |
| **Catalog** | A module that centralises all IO definitions for a project | `catalog.py` with all IO objects |
| **`run()`** | Resolves dependencies between nodes automatically and runs them in topological order | `run(pipeline_module)` |
| **`viz()`** | Generates Mermaid/Kedro-viz diagram of the pipeline graph | Mermaid diagram output |
| **Views** | Read-only nodes that don't produce outputs (for inspection/logging) | Add a view for data profiling |
| **Hooks** | Inject custom logic before/after node execution, IO load/save | Timing hook, logging hook |
| **Checks** | Validate data between nodes | Data quality check |
| **`with_load_options()` / `with_save_options()`** | Configure IO behaviour without changing code | CSV load options |
| **`with_attributes()`** | Annotate IOs with metadata (layer, description, owner) | Attribute demo |
| **Resources (`@`)** | Mark multiple IOs as using the same underlying resource | Show resource linking |
| **`pipeline()` (experimental)** | Create reusable callable pipelines | Modular sub-pipeline |
| **Custom IO** | Create your own IO class by extending `IO`/`Input`/`Output` | Custom IO for sklearn model |

---

## 🔧 Ordeq Packages We'll Use

| Package | IO Class | Purpose |
|---|---|---|
| `ordeq` | `IO`, `Input`, `Output`, `node`, `run` | Core framework |
| `ordeq-pandas` | `PandasCSV`, `PandasParquet` | Load/save DataFrames |
| `ordeq-huggingface` | `HuggingfaceDataset` | Load HF datasets |
| `ordeq-numpy` | `NumpyBinary` | Save numpy arrays |
| `ordeq-joblib` | `Joblib` | Save sklearn models |
| `ordeq-matplotlib` | `MatplotlibFigure` | Save plots |
| `ordeq-viz` | `viz()` | Pipeline visualisation |
| `ordeq-yaml` | `YAML` | Save config/results as YAML |
| `ordeq-sentence-transformers` | `SentenceTransformer` | Load sentence-transformers model |

---

## 📊 Datasets

### Pipeline 1 – Classical ML (Iris / Wine / Breast Cancer)
- **Dataset**: `scikit-learn` built-in `load_wine()` or HuggingFace `"scikit-learn/wine-quality"` or simply `sklearn.datasets`
- **Alternative (better for HF demo)**: `"mstz/heart_failure"` from HuggingFace — tabular, classification, medical, interesting
- **Best pick**: **`"scikit-learn/iris"` from HuggingFace** — universally known, simple, perfect for demo
  - Or just use `sklearn.datasets.load_iris()` saved to CSV first, then loaded via `PandasCSV`

### Pipeline 2 – LLM Text Enrichment
- **Dataset**: `"dair-ai/emotion"` from HuggingFace — text classification (6 emotions), small, fast
- **Alternative**: `"ag_news"` — news classification (4 categories)
- **Best pick**: **`"dair-ai/emotion"`** — perfect for showing text preprocessing + LLM inference
  - 6 labels: sadness, joy, love, anger, surprise, fear
  - ~20k training samples, small enough for demo

---

## 🤖 Models

### Classical ML
- **scikit-learn** `RandomForestClassifier` or `LogisticRegression`
- Saved/loaded via `Joblib` IO

### LLM / Transformer
- **HuggingFace Transformers** `pipeline("text-classification")`
  - Model: `"distilbert-base-uncased-finetuned-sst-2-english"` (sentiment) or
  - Model: `"j-hartmann/emotion-english-distilbert-roberta-base"` (emotion — matches our dataset!)
  - Or for embeddings: `SentenceTransformer("all-MiniLM-L6-v2")` via `ordeq-sentence-transformers`
- **NOT using litellm** (per your request)

### Recommended combo:
1. Use **`transformers`** library (HuggingFace) with `pipeline("text-classification", model="j-hartmann/emotion-english-distilbert-roberta-base")`
2. Use **`sentence-transformers`** for embeddings demo via Ordeq's built-in `SentenceTransformer` IO
3. Use **`scikit-learn`** for classical ML

---

## 🏗️ Project Structure

```
Symplhy-Ordeq/
├── pyproject.toml
├── SHOWCASE_PLAN.md          # this file
├── viz.py                    # generate mermaid diagrams
├── data/                     # raw + processed data lives here
│   ├── raw/
│   └── processed/
├── models/                   # saved models
├── outputs/                  # plots, metrics, results
├── src/
│   └── ordeq_showcase/
│       ├── __init__.py
│       ├── __main__.py       # entry point: run + viz
│       ├── catalog.py        # ALL IO definitions
│       ├── hooks.py          # custom hooks (timing, logging)
│       ├── custom_io.py      # custom IO demo
│       │
│       ├── classical_ml/     # Pipeline 1: Classical ML
│       │   ├── __init__.py
│       │   ├── preprocessing.py   # nodes: load data, clean, split, scale
│       │   ├── training.py        # nodes: train model
│       │   └── evaluation.py      # nodes: evaluate, confusion matrix
│       │
│       └── llm_pipeline/    # Pipeline 2: LLM Text Enrichment
│           ├── __init__.py
│           ├── data_prep.py       # nodes: load HF dataset, preprocess
│           ├── inference.py       # nodes: run model inference
│           └── analysis.py        # nodes: analyse results, metrics
│
└── tests/                    # unit tests for nodes (testing_nodes demo)
    ├── test_preprocessing.py
    └── test_inference.py
```

---

## 🔗 Key Links

| Resource | URL |
|---|---|
| Ordeq Docs - Introduction | https://ing-bank.github.io/ordeq/getting-started/introduction/ |
| Ordeq Docs - IO Concepts | https://ing-bank.github.io/ordeq/getting-started/concepts/io/ |
| Ordeq Docs - Nodes | https://ing-bank.github.io/ordeq/getting-started/concepts/nodes/ |
| Ordeq Docs - Catalogs | https://ing-bank.github.io/ordeq/getting-started/concepts/catalogs/ |
| Ordeq Docs - Views | https://ing-bank.github.io/ordeq/getting-started/concepts/views/ |
| Ordeq Docs - Checks | https://ing-bank.github.io/ordeq/getting-started/concepts/checks/ |
| Ordeq Docs - Hooks | https://ing-bank.github.io/ordeq/getting-started/concepts/hooks/ |
| Ordeq Docs - Run & Viz | https://ing-bank.github.io/ordeq/guides/run_and_viz/ |
| Ordeq Docs - Custom IO | https://ing-bank.github.io/ordeq/guides/custom_io/ |
| Ordeq Docs - Testing Nodes | https://ing-bank.github.io/ordeq/guides/testing_nodes/ |
| Ordeq Docs - Node Parameters | https://ing-bank.github.io/ordeq/guides/node_parameters/ |
| Ordeq Docs - Parametrized IO | https://ing-bank.github.io/ordeq/guides/parametrized_io/ |
| Ordeq Docs - Lazy IO | https://ing-bank.github.io/ordeq/guides/lazy_io/ |
| Ordeq Docs - Kedro comparison | https://ing-bank.github.io/ordeq/guides/kedro/ |
| GitHub Repo | https://github.com/ing-bank/ordeq |
| PyPI | https://pypi.org/project/ordeq/ |
| HF Dataset: emotion | https://huggingface.co/datasets/dair-ai/emotion |
| HF Model: emotion classifier | https://huggingface.co/j-hartmann/emotion-english-distilbert-roberta-base |
| HF Dataset: iris | https://huggingface.co/datasets/scikit-learn/iris |
| SentenceTransformers | https://www.sbert.net/ |

---

## 🗺️ Demo Flow (Feature-by-Feature)

### Part 1: Classical ML Pipeline
1. **IO & Catalog** – Define all IOs in `catalog.py` (PandasCSV, Joblib, MatplotlibFigure, NumpyBinary)
2. **`with_load_options()`** – Show CSV column type casting
3. **`with_attributes()`** – Annotate IOs with layer/description metadata
4. **Nodes** – `@node` decorator with `inputs`/`outputs` for each step
5. **Preprocessing nodes** – load → clean → feature engineer → train/test split → scale
6. **Training node** – fit model, save via Joblib
7. **Evaluation node** – accuracy, classification report → save as YAML
8. **Visualization node** – confusion matrix plot → save via MatplotlibFigure
9. **`run()`** – Execute entire pipeline
10. **`viz()`** – Generate Mermaid pipeline diagram
11. **Hooks** – Timing hook to measure node execution time
12. **Testing** – Show how to unit-test nodes by calling them directly

### Part 2: LLM Pipeline
1. **HuggingfaceDataset IO** – Load `dair-ai/emotion` dataset
2. **Custom IO** – Create a custom `TransformersModel` IO for HF pipeline
3. **Preprocessing** – Clean text, subsample for demo speed
4. **Inference node** – Run HF text-classification pipeline on dataset
5. **SentenceTransformer IO** – Generate embeddings for the texts
6. **Analysis** – Compare model predictions vs true labels
7. **Resources (`@`)** – Show two IOs pointing to same underlying data
8. **Views** – Read-only node that prints data statistics
9. **`run()` + `viz()`** – Full execution and visualisation

### Part 3: Bonus
- **`git bisect`** – Quick demo of tracking down a bug in project history

---

## 📦 Dependencies to Install

```bash
pip install ordeq ordeq-pandas ordeq-huggingface ordeq-numpy ordeq-joblib \
    ordeq-matplotlib ordeq-viz ordeq-yaml ordeq-sentence-transformers \
    scikit-learn transformers torch datasets matplotlib seaborn
```

---

## 🧠 Key Source Code Patterns (from ordeq-original)

### Basic node pattern:
```python
from ordeq import node
import catalog

@node(inputs=catalog.raw_data, outputs=catalog.clean_data)
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna()
```

### Catalog pattern:
```python
from pathlib import Path
from ordeq_pandas import PandasCSV
from ordeq_joblib import Joblib

raw_data = PandasCSV(path=Path("data/raw/iris.csv"))
clean_data = PandasCSV(path=Path("data/processed/iris_clean.csv"))
model = Joblib(path=Path("models/classifier.pkl"))
```

### Run pattern:
```python
from ordeq import run
import pipeline
run(pipeline)
```

### Viz pattern:
```python
from ordeq_viz import viz
from pathlib import Path
import pipeline
viz(pipeline, fmt="mermaid", output=Path("pipeline.mermaid"))
```

### Hook pattern:
```python
import time
from ordeq import NodeHook

class TimingHook:
    def before_node_run(self, node):
        self._start = time.time()
    
    def after_node_run(self, node):
        elapsed = time.time() - self._start
        print(f"  ⏱ {node.name}: {elapsed:.2f}s")
    
    def on_node_call_error(self, node, error):
        print(f"  ❌ {node.name} failed: {error}")
```

### Custom IO pattern:
```python
from dataclasses import dataclass
from ordeq import IO

@dataclass(frozen=True, kw_only=True)
class SklearnModel(IO):
    path: Path
    
    def load(self, **opts):
        import joblib
        return joblib.load(self.path, **opts)
    
    def save(self, model, **opts):
        import joblib
        joblib.dump(model, self.path, **opts)
```

---

## ✅ Status: Context gathered, links saved, plan ready. Waiting for you to return to start implementation.

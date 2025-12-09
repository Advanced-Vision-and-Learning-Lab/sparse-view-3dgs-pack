
# FS3DGS Benchmarking Pipeline
![Benchmarking](../image/fsgs_benchmarking.png)
The figure is adapted from <a href="https://arxiv.org/abs/2202.08341">Anomalib</a> for comparison.
---
This is the unified benchmarking pipeline for evaluating 3D Gaussian Splatting–based methods such as **3DGS**, **FSGS**, and other variants.
  
The pipeline automates:
- Training  
- Rendering  
- Metric computation (PSNR, SSIM, LPIPS)  
- Table generation (LaTeX + PNG)  
- Run-level folder organization  

---

## How to Run the Pipeline

### 1. Prepare a config file (example: `config.yaml`)

```yaml
defaults:
  dataset_root: ./datasets/mipnerf360
  out_root: ./bench_runs
  run_description: "benchmark-run"
  run_render: true
  run_metrics: true

scenes:
  - name: bicycle
    source_path: ${defaults.dataset_root}/bicycle

models:
  - name: 3dgs
    repo_dir: ./gaussian-splatting
    entry: train.py
    render_entry: render.py
    metrics_entry: metrics.py
    arg_map:
      source_path: source_path
      model_path: model_path
      sh_degree: sh_degree
      iterations: iterations
    args:
      iterations: 1000
      sh_degree: 3
````

---

## 2. Start the Benchmark

```bash
python benchmark.py -c config.yaml
```

**Dry run (only prints commands):**

```bash
python benchmark.py -c config.yaml --dry
```

---

## 3. Output Structure

Each run creates a timestamped folder inside `bench_runs/`:

```
bench_runs/
└── 2025-10-21_11-42-10_benchmark-run/
    ├── summary.csv
    ├── tables/
    │   ├── overall.{tex,png}
    │   ├── <scene>.{tex,png}
    │   ├── <model>_categories.{tex,png}
    │   └── combined.tex
    ├── <scene>__<model>__shX__itY__seedZ/
    │   ├── logs/
    │   └── model/
```

---

## 4. Tables & Figures

The pipeline automatically generates:

* Per-scene metric tables
* Per-model category tables
* Global mean ± std table
* PNG visualizations
* Combined LaTeX file

All tables are saved in:

```
bench_runs/<run_id>/tables/
```

---

## 5. Tips

* Add more models by appending to the `models:` list
* Add all scenes automatically by leaving `scenes:` empty
* Nothing is overwritten — each run gets a new timestamped folder

------------------------------------------------------------------------

# How to Add Your Own Gaussian Splatting--Based Model to This Pipeline

This guide is for those who want to integrate their **own
Gaussian Splatting--based model** into this benchmarking pipeline.\
It explains, in simple steps, how to safely add your model **without
breaking other existing models**.

This process allows **multiple GS variants to coexist** in the same
Python environment (FSGS, 3DGS, LDWTGS, custom models, etc.).

------------------------------------------------------------------------

## Why This Is Necessary

Most Gaussian Splatting repositories reuse the same package names:

-   `diff_gaussian_rasterization`
-   `simple_knn`

If two models use the same names: - The latest install **overwrites the
previous one** - Imports become ambiguous - Results become unreliable

To prevent this, **each model added to the pipeline must use a unique
name**.

------------------------------------------------------------------------

## Step-by-Step Guide for Adding Your Model

### ✅ Step 1: Copy the Model Into the Pipeline

Place your model inside the main benchmarking directory:

    fs3dgs-benchmarking/
      your_model_name/

Also copy any CUDA submodules it uses: - `diff-gaussian-rasterization` -
`simple-knn` - or similar

------------------------------------------------------------------------

### ✅ Step 2: Rename the Python Package Folder

Inside each CUDA submodule, rename the Python folder:

    diff_gaussian_rasterization  →  dgr_yourmodel
    simple_knn                  →  sknn_yourmodel

These names become your new **Python import names**.

------------------------------------------------------------------------

### ✅ Step 3: Update the `setup.py` File

Inside each renamed submodule, update three fields in `setup.py`:

``` python
setup(
    name="sknn_yourmodel",            # unique package name
    packages=["sknn_yourmodel"],      # renamed folder
    ext_modules=[
        CUDAExtension(
            name="sknn_yourmodel._C", # renamed CUDA extension
            ...
        )
    ]
)
```

Repeat this for: - `sknn_yourmodel` - `dgr_yourmodel` - any other CUDA
submodules

------------------------------------------------------------------------

### ✅ Step 4: Add a Lightweight Wrapper File

Inside each renamed folder:

    sknn_yourmodel/__init__.py

Add:

``` python
from ._C import distCUDA2
__all__ = ["distCUDA2"]
```

This allows clean imports:

``` python
from sknn_yourmodel import distCUDA2
```

------------------------------------------------------------------------

### ✅ Step 5: Avoid Duplicate Folder Names

✅ Correct:

    simple-knn-yourmodel/
      sknn_yourmodel/
        __init__.py

❌ Incorrect:

    sknn_yourmodel/
      sknn_yourmodel/

Duplicate nesting breaks Python imports.

------------------------------------------------------------------------

### ✅ Step 6: Clean and Install Your Model

From the submodule directory:

``` bash
pip uninstall -y sknn_yourmodel simple-knn simple_knn
python setup.py clean --all
rm -rf build dist *.egg-info
pip install -v .
```

Repeat for each renamed CUDA submodule.

------------------------------------------------------------------------

### ✅ Step 7: Verify That Your Model Loaded Correctly

``` python
import sknn_yourmodel, sknn_fsgs, sknn_3dgs

from sknn_yourmodel import distCUDA2
from sknn_fsgs import distCUDA2 as dist_fsgs
from sknn_3dgs import distCUDA2 as dist_3dgs

print(sknn_yourmodel.__file__)
print(sknn_fsgs.__file__)
print(sknn_3dgs.__file__)
```

If all paths print correctly, your model was added successfully.

------------------------------------------------------------------------

## Summary Checklist for New Users

Before running the benchmark, confirm:

-   ✅ You renamed the **Python package folder**
-   ✅ You updated **setup.py** with unique names
-   ✅ You added a **wrapper `__init__.py`**
-   ✅ You cleaned and reinstalled
-   ✅ You verified imports

------------------------------------------------------------------------


## Citation

If you use this code, please cite the original 3D Gaussian Splatting paper and our DWT extension:

```bibtex
@Article{kerbl3Dgaussians,
    author       = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
    title        = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
    journal      = {ACM Transactions on Graphics},
    number       = {4},
    volume       = {42},
    month        = {July},
    year         = {2023},
    url          = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}
```

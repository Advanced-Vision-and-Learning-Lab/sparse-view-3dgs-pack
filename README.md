FS3DGS-Benchmarking/
├── fs3dgs_benchmark/
│   ├── __init__.py
│   ├── cli.py
│   ├── benchmark.py
│   ├── post_install.py
│   ├── config.yaml
│   │
│   ├── FSGS/
│   │   ├── train.py
│   │   ├── render.py
│   │   ├── metrics.py
│   │   └── submodules/
│   │       ├── diff-gaussian-rasterization/
│   │       └── simple-knn/
│   │
│   ├── gaussian_splatting/
│   │   ├── train.py
│   │   ├── render.py
│   │   ├── metrics.py
│   │   └── submodules/
│   │       ├── diff-gaussian-rasterization/
│   │       ├── simple-knn/
│   │       └── fused-ssim/
│   │
│   └── DNGaussian/
│       ├── train.py
│       ├── render.py
│       ├── metrics.py
│       └── submodules/
│           ├── diff-gaussian-rasterization/
│           ├── simple-knn/
│           ├── gridencoder/
│           └── shencoder/
│
├── README.md
├── MANIFEST.in
└── setup.py

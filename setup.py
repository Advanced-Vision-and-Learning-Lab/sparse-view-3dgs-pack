from setuptools import setup, find_packages
from setuptools.command.install import install
import subprocess, pathlib

class InstallWithSubmodules(install):
    """Custom install command to build CUDA submodules after normal install."""
    def run(self):
        install.run(self)
        root = pathlib.Path(__file__).parent / "fs3dgs_benchmark"
        submods = [
            "diff-gaussian-rasterization",
            "diff-gaussian-rasterization-confidence",
            "simple-knn",
            "fused-ssim",
            "gridencoder",
            "shencoder",
        ]
        print("\n[fs3dgs_benchmark] 🔍 Building CUDA submodules…\n")
        for name in submods:
            for subdir in root.rglob(name):
                setup_py = subdir / "setup.py"
                if setup_py.exists():
                    print(f"⚙️  Building {name} at {subdir}")
                    subprocess.run(
                        ["pip", "install", ".", "--no-deps", "--no-build-isolation"],
                        cwd=subdir, check=False
                    )
from setuptools.command.develop import develop

class DevelopWithSubmodules(develop):
    """Custom develop command to also build submodules when using -e."""
    def run(self):
        develop.run(self)
        root = pathlib.Path(__file__).parent / "fs3dgs_benchmark"
        submods = [
            "diff-gaussian-rasterization",
            "diff-gaussian-rasterization-confidence",
            "simple-knn",
            "fused-ssim",
            "gridencoder",
            "shencoder",
        ]
        print("\n[fs3dgs_benchmark] 🔍 Building submodules for development...\n")
        for name in submods:
            for subdir in root.rglob(name):
                setup_py = subdir / "setup.py"
                if setup_py.exists():
                    print(f"⚙️  Building {name} in {subdir}")
                    subprocess.run(
                        ["pip", "install", ".", "--no-deps", "--no-build-isolation"],
                        cwd=subdir, check=False
                    )


with open("README.md", "r", encoding="utf-8") as f:
    readme = f.read()

setup(
    name="fs3dgs-benchmark",
    version="0.5.0",
    author="Atharva Agashe",
    author_email="atharvagashe22@tamu.edu",
    description="Unified benchmarking suite for Few-Shot 3D Gaussian Splatting methods",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/VITA-Group/FSGS",
    license="MIT",
    packages=find_packages(include=["fs3dgs_benchmark", "fs3dgs_benchmark.*"]),
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=[
        "torchmetrics",
        "numpy",
        "tqdm",
        "matplotlib",
        "open3d",
        "imageio",
        "plyfile",
        "timm",
        "opencv-python",
        "scikit-image",
        "pyyaml",
        "ninja",
    ],
    cmdclass={
    "install": InstallWithSubmodules,
    "develop": DevelopWithSubmodules,
},
    entry_points={
        "console_scripts": ["gs_benchmark = fs3dgs_benchmark.cli:main"]
    },
)


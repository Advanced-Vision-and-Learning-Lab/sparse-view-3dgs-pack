import subprocess
import pathlib
import importlib.util
from setuptools.command.build_ext import build_ext

class BuildWithSubmodules(build_ext):
    """Custom build command that installs all known CUDA submodules automatically."""

    SUBMODULE_KEYWORDS = [
        "diff-gaussian-rasterization",
        "diff-gaussian-rasterization-confidence",
        "simple-knn",
        "fused-ssim",
        "gridencoder",
        "shencoder",
    ]

    def run(self):
        super().run()
        root = pathlib.Path(__file__).resolve().parents[2]
        print("\n[fs3dgs_benchmark] 🔍 Scanning for submodules to install...\n")

        for keyword in self.SUBMODULE_KEYWORDS:
            for subdir in root.rglob(keyword):
                setup_path = subdir / "setup.py"
                if not setup_path.exists():
                    continue

                # Skip if already importable
                try:
                    if importlib.util.find_spec(keyword.replace("-", "_")):
                        print(f"[fs3dgs_benchmark] ✅ {keyword} already installed, skipping.")
                        continue
                except ModuleNotFoundError:
                    pass

                print(f"[fs3dgs_benchmark] ⚙️  Installing {keyword} at {subdir} ...")
                try:
                    subprocess.check_call(
                        ["pip", "install", ".", "--no-deps", "--no-build-isolation"],
                        cwd=str(subdir)
                    )
                except subprocess.CalledProcessError as e:
                    print(f"[fs3dgs_benchmark] ❌ Failed to build {subdir}: {e}")

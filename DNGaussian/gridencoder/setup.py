import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

_src_path = os.path.dirname(os.path.abspath(__file__))

# ... keep your imports and _src_path

nvcc_flags = [
    "-O3",
    "-std=c++17",
    "-Xcompiler", "-std=c++17",
    "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__", "-U__CUDA_NO_HALF2_OPERATORS__",

    # RTX 4090
    "-gencode=arch=compute_89,code=sm_89",
    "-gencode=arch=compute_89,code=compute_89",
    # Optional if your wheel expects it (most recent do)
    # "-D_GLIBCXX_USE_CXX11_ABI=1",
]

if os.name == "posix":
    c_flags = ["-O3", "-std=c++17"]
else:
    c_flags = ["/O2", "/std:c++17"]

setup(
    name="gridencoder",
    ext_modules=[
        CUDAExtension(
            name="_gridencoder",
            sources=[os.path.join(_src_path, "src", f) for f in [
                "gridencoder.cu",
                "bindings.cpp",
            ]],
            extra_compile_args={
                "cxx": c_flags,
                "nvcc": nvcc_flags,
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)


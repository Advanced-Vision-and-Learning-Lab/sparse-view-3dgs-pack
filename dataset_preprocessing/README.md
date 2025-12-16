# Dataset Preprocessing Pipeline

## Overview

This repository contains a modular dataset preprocessing pipeline for
generating:

-   **Full COLMAP reconstructions**
-   **Few-shot sparse and dense reconstructions**

The pipeline supports the following input sources:
-   **Standard benchmark datasets:** LLFF, Mip-NeRF 360
-   **Custom user-provided datasets:** Any folder of images placed in the required `input/` format


**Important:**\
All raw input images for each scene must be placed inside the `input/`
directory.

Example:

    datasets/
        bonsai/
            input/
                img001.jpg
                img002.jpg

------------------------------------------------------------------------

## Pipeline Flow

The following diagram illustrates the dataset pipeline workflow:

![Dataset Pipeline Flow](../assets/dataset_preprocessing_pipeline.png)

------------------------------------------------------------------------

## Usage

### Full Pipeline (Stage 1 + Stage 2)

``` bash
python data_pipeline.py     --base_path /path/to/datasets     --scene bonsai     --stage full     --n_views 10     --dataset llff
```

------------------------------------------------------------------------

### Stage 1 Only --- Full COLMAP Reconstruction

``` bash
python data_pipeline.py     --base_path /path/to/datasets     --scene bonsai     --stage part1     --dataset llff
```

------------------------------------------------------------------------

### Stage 2 Only --- Few-Shot Reconstruction

``` bash
python data_pipeline.py     --base_path /path/to/datasets     --scene bonsai     --stage part2     --n_views 10     --dataset llff
```

------------------------------------------------------------------------

## Key Arguments

### Required Arguments

-   **`--base_path`**\
    Root directory containing all scene folders.

-   **`--scene`**\
    Name of the scene folder inside the base path.

-   **`--stage`**\
    Pipeline stage to execute:\
    `full` = runs both part1 and part2\
    `part1` = full COLMAP reconstruction only\
    `part2` = few-shot reconstruction only

### Conditionally Required

-   **`--n_views`**\
    Number of views to use during few-shot reconstruction.\
    This argument is required when using `--stage part2` or
    `--stage full`.

### Optional Arguments

-   **`--dataset`**\
    Dataset type. Supported values: `llff`, `mipnerf360`, or `custom`.\
    Use `custom` when providing your own images.

-   **`--downscale`**\
    Downscale factor applied to input images before reconstruction.

-   **`--max_num_features`**\
    Manually override the default SIFT feature limit.

------------------------------------------------------------------------

## Output Structure

After running the pipeline, your scene directory will look like:

    scene/
        input/
        images/
        sparse/0/
        poses_bounds.npy
        fewshot/
            n_views10/
                sparse/
                dense/

------------------------------------------------------------------------

## Notes

-   Raw images must always be placed inside `input/` before running the
    pipeline.
-   You may use either **standard datasets (LLFF, Mip-NeRF 360)** or
    **your own custom image collections**.
-   SIFT features are automatically selected:
    -   LLFF → 32768 features
    -   Mip-NeRF 360 → 16384 features
-   Pipeline stages are automatically skipped if outputs already exist.
-   `part2` requires both `images/` and `sparse/0/` to be present.

------------------------------------------------------------------------

## Supported Standard Datasets and Citations

This project relies on publicly available benchmark datasets.\
If you use any of the following datasets, please cite the corresponding
papers.

------------------------------------------------------------------------

### LLFF (Local Light Field Fusion)

**Paper:**\
Local Light Field Fusion: Practical View Synthesis with Prescriptive
Sampling Guidelines\
Ben Mildenhall et al., ACM TOG 2019

**BibTeX:**

``` bibtex
@article{mildenhall2019local,
  title={Local light field fusion: Practical view synthesis with prescriptive sampling guidelines},
  author={Mildenhall, Ben and Srinivasan, Pratul P and Ortiz-Cayon, Rodrigo and Kalantari, Nima Khademi and Ramamoorthi, Ravi and Ng, Ren and Kar, Abhishek},
  journal={ACM Transactions on Graphics (ToG)},
  volume={38},
  number={4},
  pages={1--14},
  year={2019},
  publisher={ACM New York, NY, USA}
}
```

------------------------------------------------------------------------

### Mip-NeRF 360

**Paper:**\
Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields\
Jonathan T. Barron et al., CVPR 2022

**BibTeX:**

``` bibtex
@inproceedings{barron2022mip,
  title={Mip-nerf 360: Unbounded anti-aliased neural radiance fields},
  author={Barron, Jonathan T and Mildenhall, Ben and Verbin, Dor and Srinivasan, Pratul P and Hedman, Peter},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={5470--5479},
  year={2022}
}
```

------------------------------------------------------------------------

## Code Acknowledgements

This project builds upon and adapts implementations from the following
open-source research repositories:

------------------------------------------------------------------------

### 3D Gaussian Splatting

**GitHub:**\
https://github.com/graphdeco-inria/gaussian-splatting

------------------------------------------------------------------------

### FSGS (Few-Shot Gaussian Splatting)

**GitHub:**\
https://github.com/VITA-Group/FSGS/tree/main

------------------------------------------------------------------------

### LLFF Codebase

**GitHub:**\
https://github.com/Fyusion/LLFF

------------------------------------------------------------------------

## Dataset and Code Usage Disclaimer

This repository contains original code and adaptations based on publicly
available research implementations. All third-party datasets, methods,
and code remain the property of their respective authors and are used in
accordance with their licenses.

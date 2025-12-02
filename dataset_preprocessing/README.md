# Dataset Pipeline

## Overview

This repository contains a modular dataset preprocessing pipeline for
generating **full COLMAP reconstructions** and **few-shot sparse/dense
reconstructions**.\
The pipeline supports **LLFF** and **MipNeRF360** datasets, includes
dataset-aware SIFT feature selection, and automatically skips stages
that have already been completed.

**Important:**\
All raw input images for each scene must be placed inside the `input/`
folder:

    datasets/
        bonsai/
            input/
                img001.jpg
                img002.jpg

------------------------------------------------------------------------

## Pipeline Flow

The following diagram illustrates the dataset pipeline workflow:

![Dataset Pipeline Flow](docs/pipeline_flow.png)

*(Place your flowchart image as `docs/pipeline_flow.png`.)*

------------------------------------------------------------------------

## Usage

### Full Pipeline (Stage 1 + Stage 2)

``` bash
python data_pipeline.py     --base_path /path/to/datasets     --scene bonsai     --stage full     --n_views 10     --dataset llff
```

### Stage 1 Only --- Full COLMAP Reconstruction

``` bash
python data_pipeline.py     --base_path /path/to/datasets     --scene bonsai     --stage part1     --dataset llff
```

### Stage 2 Only --- Few-Shot Reconstruction

``` bash
python data_pipeline.py     --base_path /path/to/datasets     --scene bonsai     --stage part2     --n_views 10     --dataset llff
```

------------------------------------------------------------------------

## Key Arguments

  Argument               Description                            Required
  ---------------------- -------------------------------------- ---------------------------
  `--base_path`          Root directory containing all scenes   Yes
  `--scene`              Scene folder name                      Yes
  `--stage`              `full`, `part1`, `part2`               Yes
  `--n_views`            Views for few-shot stage               Required for `part2/full`
  `--dataset`            `llff` or `mipnerf360`                 Recommended
  `--downscale`          Image downscale factor                 Optional
  `--max_num_features`   Override SIFT limit                    Optional

------------------------------------------------------------------------

## Output Structure

After running the pipeline, your directory should look like:

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

-   Raw images must always be inside `input/` before running.
-   SIFT features auto-selected: LLFF → 32768, MipNeRF360 → 16384.
-   Stages are skipped if outputs already exist.
-   `part2` requires `images/` + `sparse/0/`.

------------------------------------------------------------------------

## Citation

--
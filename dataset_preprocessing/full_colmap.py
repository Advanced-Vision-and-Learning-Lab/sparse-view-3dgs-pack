"""
Adapted from convert.py in the 3D Gaussian Splatting (3DGS) repository.
Original work © 2023 Inria, GRAPHDECO research group.
Released for non-commercial research use under the LICENSE.md of 3DGS.

This module implements Stage 1 of the pipeline:
- COLMAP feature extraction
- Matching
- Sparse reconstruction
- Undistortion

Refactored into a reusable function (run_full_colmap) for modular
integration in the FS3DGS-Benchmarking few-shot pipeline.
"""



import os
import logging
import shutil
import subprocess
import sys

from poses.pose_utils import gen_poses


def run_cmd(cmd: str):
    print(f"[CMD] {cmd}")
    exit_code = os.system(cmd)
    if exit_code != 0:
        logging.error(f"Command failed with code {exit_code}: {cmd}")
        sys.exit(exit_code)

def run_cmd_with_error(cmd: str):
    print(f"[CMD] {cmd}")
    exit_code = os.system(cmd)
    if exit_code != 0:
        logging.error(f"Command failed with code {exit_code}: {cmd}")
        raise subprocess.CalledProcessError(exit_code, cmd)


def run_full_colmap(scene_root: str, input_folder: str, colmap_command: str, magick_command: str,
                    camera_model: str, use_gpu: bool, skip_matching: bool,
                    resize: bool):
    """
    Run the full COLMAP reconstruction pipeline on a scene.

    Inputs:
    - scene_root: The path to the scene directory containing `input/`
    - colmap_command: Path or alias to COLMAP executable
    - magick_command: ImageMagick executable
    - camera_model: Camera model (e.g., OPENCV)
    - use_gpu: Whether to use GPU for SIFT
    - skip_matching: Skip feature extraction + matching (if DB already exists)
    - resize: Whether to generate downscaled image pyramids

    Output folders created:
    - scene_root/distorted/sparse/0/
    - scene_root/images/
    - scene_root/sparse/0/
    """

    distorted_sparse_dir = os.path.join(scene_root, "distorted", "sparse")
    input_dir = os.path.join(scene_root, input_folder)

    if not skip_matching:
        os.makedirs(distorted_sparse_dir, exist_ok=True)

        # Feature extraction
        db_path = os.path.join(scene_root, "distorted", "database.db")
        feat_cmd = (
            f"{colmap_command} feature_extractor "
            f"--database_path {db_path} "
            f"--image_path {input_dir} "
            f"--ImageReader.single_camera 1 "
            f"--ImageReader.camera_model {camera_model} "
            f"--SiftExtraction.use_gpu {int(use_gpu)}"
        )
        run_cmd(feat_cmd)

        # Matching
        match_cmd = (
            f"{colmap_command} exhaustive_matcher "
            f"--database_path {db_path} "
            f"--SiftMatching.use_gpu {int(use_gpu)}"
        )
        run_cmd(match_cmd)

        # # Mapper
        # mapper_cmd = (
        #     f"{colmap_command} mapper "
        #     f"--database_path {db_path} "
        #     f"--image_path {input_dir} "
        #     f"--output_path {distorted_sparse_dir} "
        #     f"--Mapper.ba_global_function_tolerance=0.000001"
        # )
        # run_cmd(mapper_cmd)
        # Mapper (first attempt)
        mapper_cmd = (
            f"{colmap_command} mapper "
            f"--database_path {db_path} "
            f"--image_path {input_dir} "
            f"--output_path {distorted_sparse_dir} "
            f"--Mapper.ba_global_function_tolerance=0.000001"
        )

        try:
            run_cmd_with_error(mapper_cmd)
        except subprocess.CalledProcessError as e:
            print("Mapper failed without min_tri_angle. Retrying with init_min_tri_angle=4 ...")
            print("Error was:", e)

            mapper_cmd_retry = (
                f"{colmap_command} mapper "
                f"--database_path {db_path} "
                f"--image_path {input_dir} "
                f"--output_path {distorted_sparse_dir} "
                f"--Mapper.ba_global_function_tolerance=0.000001 "
                f"--Mapper.init_min_tri_angle 4"
            )

            run_cmd(mapper_cmd_retry)


    
        
    # Undistortion
    distorted_sparse0 = os.path.join(distorted_sparse_dir, "0")
    undist_cmd = (
        f"{colmap_command} image_undistorter "
        f"--image_path {input_dir} "
        f"--input_path {distorted_sparse0} "
        f"--output_path {scene_root} "
        f"--output_type COLMAP"
    )
    run_cmd(undist_cmd)

    # Move sparse output into sparse/0
    sparse_root = os.path.join(scene_root, "sparse")
    os.makedirs(os.path.join(sparse_root, "0"), exist_ok=True)

    for f in os.listdir(sparse_root):
        if f == "0":
            continue
        src = os.path.join(sparse_root, f)
        dst = os.path.join(sparse_root, "0", f)
        if os.path.isfile(src):
            shutil.move(src, dst)

    # Optional resize
    if resize:
        imgs_dir = os.path.join(scene_root, "images")
        for scale, factor in [("images_2", "50%"), ("images_4", "25%"), ("images_8", "12.5%")]:
            out_dir = os.path.join(scene_root, scale)
            os.makedirs(out_dir, exist_ok=True)
            for img in os.listdir(imgs_dir):
                src = os.path.join(imgs_dir, img)
                dst = os.path.join(out_dir, img)
                shutil.copy2(src, dst)
                run_cmd(f"{magick_command} mogrify -resize {factor} {dst}")

    
    print("[FULL COLMAP] Completed.")

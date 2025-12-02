"""
data_pipeline.py

This script orchestrates the modular pipeline:
- Stage 1: Full COLMAP reconstruction
- Stage 2: Few-shot reconstruction

User can select:
--stage full   → run full + few-shot
--stage part1  → run full only
--stage part2  → run few-shot only

Includes dataset-aware feature count selection.
"""

import os
import sys
import logging
from argparse import ArgumentParser

from poses.pose_utils import gen_poses
from full_colmap import run_full_colmap
from fewshot_colmap import run_fewshot_pipeline


def determine_default_features(dataset: str) -> int:
    """
    Determine max_num_features based on dataset type.

    - LLFF → 32768
    - MipNeRF360 → 16384
    """
    if dataset == "mipnerf360":
        return 16384
    return 32768

def folder_exists(path):
    return os.path.exists(path) and os.path.isdir(path)


def full_stage_output_exists(scene_root):
    images_ok = folder_exists(os.path.join(scene_root, "images"))
    sparse_ok = folder_exists(os.path.join(scene_root, "sparse", "0"))
    poses_ok = os.path.exists(os.path.join(scene_root, "poses_bounds.npy"))
    return images_ok and sparse_ok, poses_ok

def main():
    parser = ArgumentParser("Modular Few-Shot COLMAP Pipeline")

    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument("--scene", type=str, required=True)
    parser.add_argument("--input_folder", type=str, default="input", required=False)
    parser.add_argument("--stage", type=str, required=True,
                        choices=["full", "part1", "part2"],
                        help="Select pipeline stage to run")

    parser.add_argument("--n_views", type=int, default=None,
                        help="Number of views for few-shot stage")

    parser.add_argument("--max_num_features", type=int, default=None,
                        help="Override SIFT feature limit; if not set, dataset defaults apply")

    parser.add_argument("--dataset", type=str, default=None,
                        choices=["llff", "mipnerf360"],
                        help="Dataset type for auto feature settings")
    parser.add_argument(
                        "--downscale",type=int,default=1,
                        help="Image downscale factor: 1 (images), 2 (images_2), 4 (images_4), 8 (images_8). "
                            "If omitted, uses dataset-based default (images for LLFF, images_8 for MipNeRF360).")

    parser.add_argument("--no_gpu", action="store_true")
    parser.add_argument("--skip_matching", action="store_true")
    parser.add_argument("--camera", default="OPENCV")
    parser.add_argument("--colmap_executable", default="")
    parser.add_argument("--magick_executable", default="magick")
    parser.add_argument("--resize", action="store_true")

    args = parser.parse_args()

    # -----------------------
    # Auto-set SIFT max features
    # -----------------------
    if args.max_num_features is None:
        args.max_num_features = determine_default_features(args.dataset)

    print(f"[INFO] Using max_num_features = {args.max_num_features}")

    # Tool commands
    colmap_cmd = f'"{args.colmap_executable}"' if args.colmap_executable else "colmap"
    magick_cmd = f'"{args.magick_executable}"'

    use_gpu = not args.no_gpu

    scene_root = os.path.join(args.base_path, args.scene)
    input_folder = args.input_folder
    logging.basicConfig(level=logging.INFO)

    print(f"[INFO] Scene root: {scene_root}")
    print(f"[INFO] Stage: {args.stage}")

    # ----------------------
    # PART 1 ONLY
    # ----------------------
    if args.stage == "part1":
        colmap_recons, poses_ok =  full_stage_output_exists(scene_root)
        if colmap_recons:
            if poses_ok:
                print("[INFO] Stage 1 output already exists. Nothing to do.")
                return
            else:
                gen_poses(scene_root, "exhaustive_matcher")
        else:        
            run_full_colmap(scene_root, input_folder, colmap_cmd, magick_cmd,
                            args.camera, use_gpu,
                            args.skip_matching, args.resize)
        print("[DONE] Part 1 completed.")
        return

    # ----------------------
    # PART 2 ONLY
    # ----------------------
    if args.stage == "part2":
        if not os.path.exists(os.path.join(scene_root, "sparse", "0")):
            print("ERROR: sparse/0 missing — cannot run part2.")
            sys.exit(1)
        if not os.path.exists(os.path.join(scene_root, "images")):
            print("ERROR: images/ missing — cannot run part2.")
            sys.exit(1)
        if args.n_views is None:
            print("ERROR: --n_views required for part2.")
            sys.exit(1)

        run_fewshot_pipeline(
            scene_root=scene_root,
            colmap_command=colmap_cmd,
            n_views=args.n_views,
            max_num_features=args.max_num_features,
            dataset=args.dataset,
            downscale=args.downscale,
        )

        print("[DONE] Part 2 completed.")
        return

    # ----------------------
    # FULL PIPELINE
    # ----------------------
    if args.stage == "full":
        colmap_recons, poses_ok =  full_stage_output_exists(scene_root)
        if colmap_recons:
            if poses_ok:
                print("[INFO] Stage 1 output already exists. Nothing to do.")
                return
            else:
                print("[INFO] poses_bounds.npy file doesn exist output already exists. Nothing to do.")
                print("[RUNNING] Generate Poses")
                gen_poses(scene_root, "exhaustive_matcher")
        else:
            print("[RUNNING] Stage 1: Full reconstruction...")        
            run_full_colmap(scene_root, input_folder, colmap_cmd, magick_cmd,
                            args.camera, use_gpu,
                            args.skip_matching, args.resize)


        print("[RUNNING] Stage 2: Few-shot reconstruction...")
        run_fewshot_pipeline(
            scene_root=scene_root,
            colmap_command=colmap_cmd,
            n_views=args.n_views,
            max_num_features=args.max_num_features,
            dataset=args.dataset,
            downscale=args.downscale,
        )


        print("[DONE] Full pipeline completed.")
        return


if __name__ == "__main__":
    main()

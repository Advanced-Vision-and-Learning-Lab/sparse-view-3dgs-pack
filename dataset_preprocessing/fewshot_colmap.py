"""
fewshot_colmap.py

Adapted from colmap_llff.py and colmap_360.py in the FSGS repository.
Original work © authors of FSGS, for research use.

Implements Stage 2 (Few-Shot Reconstruction):
- Read camera model from sparse/0/cameras.txt
- Select n_views images
- Choose resolution folder (images, images_2, images_4, images_8)
- Extract COLMAP features using the SAME camera model as Stage 1
- Matching, triangulation, undistortion
- Dense stereo + fusion

Refactored for modular integration in an FS3DGS-style benchmarking pipeline.
"""

import os
import sys
import shutil
import sqlite3
from typing import Optional
import numpy as np

from full_colmap import run_cmd  # assumes full_colmap.py is in the same package

# -------------------------------------------------------------------------
# SQLite schema helpers (from original FSGS scripts; minimal subset kept)
# -------------------------------------------------------------------------

IS_PYTHON3 = sys.version_info[0] >= 3
MAX_IMAGE_ID = 2**31 - 1

CREATE_CAMERAS_TABLE = """CREATE TABLE IF NOT EXISTS cameras (
    camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    model INTEGER NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    params BLOB,
    prior_focal_length INTEGER NOT NULL)"""

CREATE_IMAGES_TABLE = """CREATE TABLE IF NOT EXISTS images (
    image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    name TEXT NOT NULL UNIQUE,
    camera_id INTEGER NOT NULL,
    prior_qw REAL,
    prior_qx REAL,
    prior_qy REAL,
    prior_qz REAL,
    prior_tx REAL,
    prior_ty REAL,
    prior_tz REAL,
    CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < {}),
    FOREIGN KEY(camera_id) REFERENCES cameras(camera_id))""".format(MAX_IMAGE_ID)

CREATE_KEYPOINTS_TABLE = """CREATE TABLE IF NOT EXISTS keypoints (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)"""

CREATE_DESCRIPTORS_TABLE = """CREATE TABLE IF NOT EXISTS descriptors (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)"""

CREATE_MATCHES_TABLE = """CREATE TABLE IF NOT EXISTS matches (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB)"""

CREATE_TWO_VIEW_GEOMETRIES_TABLE = """CREATE TABLE IF NOT EXISTS two_view_geometries (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    config INTEGER NOT NULL,
    F BLOB,
    E BLOB,
    H BLOB,
    qvec BLOB,
    tvec BLOB)"""

CREATE_NAME_INDEX = "CREATE UNIQUE INDEX IF NOT EXISTS index_name ON images(name)"

CREATE_ALL = "; ".join([
    CREATE_CAMERAS_TABLE,
    CREATE_IMAGES_TABLE,
    CREATE_KEYPOINTS_TABLE,
    CREATE_DESCRIPTORS_TABLE,
    CREATE_MATCHES_TABLE,
    CREATE_TWO_VIEW_GEOMETRIES_TABLE,
    CREATE_NAME_INDEX,
])


def array_to_blob(a: np.ndarray) -> bytes:
    return a.tobytes() if IS_PYTHON3 else np.getbuffer(a)


def round_python3(number: float) -> float:
    """
    Python 3 round() is banker's rounding; this helper reproduces the
    legacy behavior used in some older code paths.
    """
    rounded = round(number)
    if abs(number - rounded) == 0.5:
        return 2.0 * round(number / 2.0)
    return rounded


class COLMAPDatabase(sqlite3.Connection):
    @staticmethod
    def connect(path: str) -> "COLMAPDatabase":
        return sqlite3.connect(path, factory=COLMAPDatabase)

    def __init__(self, *args, **kwargs):
        super(COLMAPDatabase, self).__init__(*args, **kwargs)
        self.create_tables = lambda: self.executescript(CREATE_ALL)


# -------------------------------------------------------------------------
# Helpers: camera model + resolution folder
# -------------------------------------------------------------------------

def read_camera_model(cameras_txt_path: str) -> str:
    """
    Read the camera model from sparse/0/cameras.txt.

    Example line:
        1 PINHOLE 4946 3286 fx fy cx cy
    Returns:
        model string, e.g. "PINHOLE", "OPENCV", "SIMPLE_RADIAL".
    """
    with open(cameras_txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            # CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]
            return parts[1]
    raise RuntimeError("No valid camera entry found in cameras.txt")

def resolve_image_folder(scene_root: str,
                         dataset: Optional[str],
                         downscale: Optional[int]) -> str:

    """
    Decide which image folder to use for few-shot images.

    Priority:
    1) If downscale is provided:
         - downscale == 1 → 'images'
         - downscale == k → f'images_{k}'
       Error if that folder does not exist.

    2) If dataset == 'mipnerf360':
         Prefer 'images_8', then 'images'.

    3) For dataset == 'llff' or None:
         Prefer 'images', then 'images_8'.

    Returns:
        Absolute path to chosen image folder.
    """
    # Explicit user choice via downscale
    if downscale is not None:
        if downscale == 1:
            folder_name = "images"
        else:
            folder_name = f"images_{downscale}"
        path = os.path.join(scene_root, folder_name)
        if not os.path.isdir(path):
            raise FileNotFoundError(
                f"Requested downscale={downscale} but folder '{folder_name}' "
                f"does not exist under {scene_root}"
            )
        return path

    # Automatic: based on dataset type
    candidates: list[str]
    if dataset == "mipnerf360":
        candidates = [
            os.path.join(scene_root, "images_8"),
            os.path.join(scene_root, "images"),
        ]
    else:  # llff or unknown → original LLFF behavior
        candidates = [
            os.path.join(scene_root, "images"),
            os.path.join(scene_root, "images_8"),
        ]

    for path in candidates:
        if os.path.isdir(path):
            return path

    raise FileNotFoundError(
        f"Could not find a suitable image folder under {scene_root}. "
        f"Tried: {candidates}"
    )


# -------------------------------------------------------------------------
# Few-Shot Pipeline
# -------------------------------------------------------------------------

def run_fewshot_pipeline(
    scene_root: str,
    colmap_command: str,
    n_views: int,
    max_num_features: int,
    dataset: Optional[str] = None,
    downscale: Optional[int] = None,

) -> None:
    """
    Run few-shot reconstruction using the full sparse model.

    Args:
        scene_root: Path to the scene directory (contains sparse/0, images, etc.).
        colmap_command: Name or path to the COLMAP executable.
        n_views: Number of views to keep for the few-shot subset.
        max_num_features: SIFT feature upper bound for extraction/matching.
        dataset: Optional dataset type; supported: 'llff', 'mipnerf360'.
                 Affects default resolution folder when downscale is None.
        downscale: Optional desired downscale factor:
                   - 1  → use 'images'
                   - 2  → use 'images_2'
                   - 4  → use 'images_4'
                   - 8  → use 'images_8'
                   If None, auto-select based on dataset.

    Output:
        Creates a folder:
            scene_root / f"{n_views}_views" /
                created/
                triangulated/
                images/
                dense/
    """

    llffhold = 8
    view_dir = os.path.join(scene_root, f"{n_views}_views")

    # Clean workspace
    if os.path.exists(view_dir):
        shutil.rmtree(view_dir)
    os.makedirs(view_dir)
    os.makedirs(os.path.join(view_dir, "created"))
    os.makedirs(os.path.join(view_dir, "triangulated"))
    os.makedirs(os.path.join(view_dir, "images"))

    # Convert parent model to TXT (in case it's still in binary)
    sparse0 = os.path.join(scene_root, "sparse", "0")
    run_cmd(
        f"{colmap_command} model_converter "
        f"--input_path {sparse0} --output_path {sparse0} --output_type TXT"
    )

    # Read camera model to ensure consistency (PINHOLE, SIMPLE_RADIAL, etc.)
    cameras_txt = os.path.join(sparse0, "cameras.txt")
    camera_model = read_camera_model(cameras_txt)
    print(f"[INFO] Camera model detected for few-shot: {camera_model}")

    # Parse images.txt from full reconstruction
    images_dict: dict[str, list[str]] = {}
    images_file = os.path.join(sparse0, "images.txt")
    with open(images_file, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and not line.startswith("#"):
                elems = line.split()
                image_name = elems[9]
                # second line (2D points) is ignored
                _ = fid.readline()
                images_dict[image_name] = elems[1:]  # store [q, t, camera_id, name, ...] tail

    # LLFF-style holdout + subsampling to n_views
    all_imgs = sorted(images_dict.keys())
    train_imgs = [c for idx, c in enumerate(all_imgs) if idx % llffhold != 0]

    if n_views > 0 and len(train_imgs) > 0:
        idx_sub = [round_python3(i) for i in np.linspace(0, len(train_imgs) - 1, n_views)]
        idx_sub = sorted(set(int(i) for i in idx_sub))
        train_imgs = [c for idx, c in enumerate(train_imgs) if idx in idx_sub]

    # Choose base image folder (resolution)
    base_images_folder = resolve_image_folder(scene_root, dataset, downscale)
    print(f"[INFO] Using image folder for few-shot: {base_images_folder}")

    # Copy selected images into few-shot workspace
    for img in train_imgs:
        src = os.path.join(base_images_folder, img)
        dst = os.path.join(view_dir, "images", img)
        if not os.path.isfile(src):
            raise FileNotFoundError(f"Expected image '{src}' not found.")
        shutil.copy2(src, dst)

    # Prepare created model: copy cameras and create empty points3D
    shutil.copy2(
        os.path.join(sparse0, "cameras.txt"),
        os.path.join(view_dir, "created", "cameras.txt"),
    )
    with open(os.path.join(view_dir, "created", "points3D.txt"), "w"):
        pass

    # ---------------------------------------------------------------------
    # Feature extraction + matching (consistent camera model)
    # ---------------------------------------------------------------------
    old_cwd = os.getcwd()
    os.chdir(view_dir)

    db_path = "database.db"

    # Feature extraction
    feat_cmd = (
        f"{colmap_command} feature_extractor "
        f"--database_path {db_path} "
        f"--image_path images "
        f"--ImageReader.camera_model {camera_model} "
        f"--ImageReader.single_camera 1 "
        f"--SiftExtraction.max_image_size 4032 "
        f"--SiftExtraction.max_num_features {max_num_features} "
        f"--SiftExtraction.estimate_affine_shape 1 "
        f"--SiftExtraction.domain_size_pooling 1"
    )
    run_cmd(feat_cmd)

    # Matching
    match_cmd = (
        f"{colmap_command} exhaustive_matcher "
        f"--database_path {db_path} "
        f"--SiftMatching.guided_matching 1 "
        f"--SiftMatching.max_num_matches {max_num_features}"
    )
    run_cmd(match_cmd)

    # Load images from DB to ensure ordering matches COLMAP
    db = COLMAPDatabase.connect(db_path)
    db_images = db.execute("SELECT * FROM images")
    img_rank = [rec[1] for rec in db_images]  # rec[1] is 'name'

    # Write created/images.txt in the new image order
    created_images_txt = os.path.join("created", "images.txt")
    with open(created_images_txt, "w") as fid:
        for idx, img_name in enumerate(img_rank):
            base_name = os.path.basename(img_name)
            if base_name not in images_dict:
                raise KeyError(
                    f"Image '{base_name}' from DB not found in original images.txt mapping."
                )
            fid.write(str(1 + idx))
            for item in images_dict[base_name]:
                fid.write(" " + item)
            fid.write("\n\n")

    # ---------------------------------------------------------------------
    # Triangulation + dense reconstruction
    # ---------------------------------------------------------------------
    tri_cmd = (
        f"{colmap_command} point_triangulator "
        f"--database_path {db_path} "
        f"--image_path images "
        f"--input_path created "
        f"--output_path triangulated "
        f"--Mapper.ba_local_max_num_iterations 40 "
        f"--Mapper.ba_local_max_refinements 3 "
        f"--Mapper.ba_global_max_num_iterations 100"
    )
    run_cmd(tri_cmd)

    run_cmd(
        f"{colmap_command} model_converter "
        f"--input_path triangulated "
        f"--output_path triangulated "
        f"--output_type TXT"
    )

    run_cmd(
        f"{colmap_command} image_undistorter "
        f"--image_path images "
        f"--input_path triangulated "
        f"--output_path dense"
    )

    run_cmd(
        f"{colmap_command} patch_match_stereo "
        f"--workspace_path dense"
    )

    run_cmd(
        f"{colmap_command} stereo_fusion "
        f"--workspace_path dense "
        f"--output_path dense/fused.ply"
    )

    os.chdir(old_cwd)
    print(f"[FEWSHOT] Completed few-shot reconstruction for n_views={n_views}.")

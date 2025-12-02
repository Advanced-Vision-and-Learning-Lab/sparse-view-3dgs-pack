# preprocess_handler.py
import os
import subprocess
import logging

class DatasetCompatibilityError(Exception):
    """Raised when dataset structure is incompatible with model requirements."""
    pass


def check_dataset_compatibility(scene_path: str, model_name: str) -> bool:
    """
    Performs a lightweight dataset sanity check.
    You can expand this logic later for model-specific dataset validation.
    """
    if not os.path.exists(scene_path):
        raise DatasetCompatibilityError(f"Scene path not found: {scene_path}")

    # Simple heuristic check (extendable per model)
    expected_items = ["images", "sparse"]
    missing = [item for item in expected_items if not os.path.exists(os.path.join(scene_path, item))]

    if missing:
        logging.warning(f"[{model_name}] Missing items in dataset: {missing}")
        return False
    return True


def get_converted_path(defaults: dict, model_name: str, scene_cfg: dict) -> str:
    """
    Returns the output path for the converted dataset following:
    data/<model>/<dataset>/<scene>/
    """
    dataset_name = os.path.basename(defaults["dataset_root"].rstrip("/"))
    scene_name = scene_cfg["name"]
    converted_root = os.path.join("data", model_name, dataset_name, scene_name)
    os.makedirs(converted_root, exist_ok=True)
    return converted_root


def run_preprocessing(model_cfg: dict, scene_cfg: dict, defaults: dict, converted_path: str):
    """
    Executes the model-specific preprocessing script.
    """
    preprocess_script = model_cfg.get("preprocess_entry")
    if not preprocess_script:
        raise ValueError(f"No preprocess_entry specified for model {model_cfg['name']}")

    repo_dir = model_cfg["repo_dir"]
    script_path = os.path.join(repo_dir, preprocess_script)
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Preprocessing script not found: {script_path}")

    # Build command
    cmd = ["python", script_path, "-s", scene_cfg["source_path"], "-o", converted_path]
    preprocess_args = model_cfg.get("preprocess_args", {})

    for k, v in preprocess_args.items():
        if isinstance(v, bool):
            if v:
                cmd.append(f"--{k}")
        else:
            cmd.extend([f"--{k}", str(v)])

    logging.info(f"[Preprocessing] Running: {' '.join(cmd)}")

    # Execute
    result = subprocess.run(cmd, cwd=repo_dir, capture_output=True, text=True)

    if result.returncode != 0:
        logging.error(f"[Preprocessing] Error while running {script_path}: {result.stderr}")
        raise RuntimeError(f"Preprocessing failed for {model_cfg['name']}:{scene_cfg['name']}")
    else:
        logging.info(f"[Preprocessing] Conversion completed successfully for {model_cfg['name']}:{scene_cfg['name']}")


def handle_dataset_preprocessing(model_cfg: dict, scene_cfg: dict, defaults: dict) -> str:
    """
    Main entry point:
    1. Checks dataset compatibility.
    2. Runs preprocessing if required.
    3. Updates and returns scene_cfg["source_path"].
    """
    model_name = model_cfg["name"]
    original_path = scene_cfg["source_path"]

    # logging.info(f"[Preprocessing] Checking dataset for {model_name}:{scene_cfg['name']} ...")

    # preprocess_flag = model_cfg.get("preprocess_dataset", defaults.get("preprocess_dataset", False))

    # try:
    #     compatible = check_dataset_compatibility(original_path, model_name)
    # except DatasetCompatibilityError as e:
    #     logging.warning(f"[Preprocessing] {e}")
    #     compatible = False

    # # If dataset incompatible or explicit flag set, trigger conversion
    # if not compatible or preprocess_flag:
    #     converted_path = get_converted_path(defaults, model_name, scene_cfg)
    #     run_preprocessing(model_cfg, scene_cfg, defaults, converted_path)
    #     scene_cfg["source_path"] = converted_path
    #     logging.info(f"[Preprocessing] Updated path for {model_name}:{scene_cfg['name']} → {converted_path}")
    # else:
    #     logging.info(f"[Preprocessing] Dataset for {model_name}:{scene_cfg['name']} is compatible. Using original path.")

    return scene_cfg["source_path"]

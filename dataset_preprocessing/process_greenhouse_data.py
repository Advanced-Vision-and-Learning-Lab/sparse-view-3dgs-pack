import os, shutil
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from itertools import product

# ------------------------------------------------------------
# Utility
# ------------------------------------------------------------

def ensure(path):
    os.makedirs(path, exist_ok=True)

def tile(image, d=2):
    w, h = image.size
    grid = product(range(0, h - h % d, d), range(0, w - w % d, d))
    return [(j, i, j + d, i + d) for i, j in grid]

def get_channels(image_path):
    raw_image = Image.open(image_path)
    size = raw_image.size[0] // 2
    slices = tile(raw_image, d=size)

    img_stack = np.zeros((size, size, len(slices)))
    for idx, box in enumerate(slices):
        img_stack[:, :, idx] = np.array(raw_image.crop(box))

    red      = np.expand_dims(img_stack[:, :, 1], axis=-1)
    green    = np.expand_dims(img_stack[:, :, 0], axis=-1)
    red_edge = np.expand_dims(img_stack[:, :, 2], axis=-1)
    nir      = np.expand_dims(img_stack[:, :, -1], axis=-1)

    return red, green, red_edge, nir

def normalize(channel):
    ch = (channel - channel.min()) / (channel.max() - channel.min() + 1e-8)
    return (ch * 255).astype(np.uint8).squeeze()


# ------------------------------------------------------------
# Step 1 — Merge TIFFs
# ------------------------------------------------------------

def merge_tiffs(scene_path, scene_name):
    cam1 = os.path.join(scene_path, "TIFF", "Camera1")
    cam2 = os.path.join(scene_path, "TIFF", "Camera2")

    out_dir = os.path.join(scene_path, f"{scene_name}_merged")
    ensure(out_dir)

    VALID_EXT = {".tif", ".tiff"}

    def is_img(p, n):
        return os.path.isfile(os.path.join(p, n)) and os.path.splitext(n)[1].lower() in VALID_EXT

    files1 = {f for f in os.listdir(cam1) if is_img(cam1, f)}
    files2 = {f for f in os.listdir(cam2) if is_img(cam2, f)}

    both = sorted(files1 & files2)
    only1 = sorted(files1 - files2)
    only2 = sorted(files2 - files1)

    for f in both:
        name, ext = os.path.splitext(f)
        shutil.copyfile(os.path.join(cam1, f), os.path.join(out_dir, f"{name}__cam1{ext}"))
        shutil.copyfile(os.path.join(cam2, f), os.path.join(out_dir, f"{name}__cam2{ext}"))

    for f in only1:
        name, ext = os.path.splitext(f)
        shutil.copyfile(os.path.join(cam1, f), os.path.join(out_dir, f"{name}__cam1{ext}"))

    for f in only2:
        name, ext = os.path.splitext(f)
        shutil.copyfile(os.path.join(cam2, f), os.path.join(out_dir, f"{name}__cam2{ext}"))

    print(f"✔ Merged TIFFs for {scene_name}")
    return out_dir


# ------------------------------------------------------------
# Step 2 — Generate RGB composites
# ------------------------------------------------------------

def generate_rgb(scene_path, scene_name, merged_dir):
    out_rgb = os.path.join(scene_path, f"{scene_name}_rgb")
    ensure(out_rgb)

    imgs = [f for f in os.listdir(merged_dir) if f.lower().endswith(('.tif', '.tiff'))]

    for img in imgs:
        path = os.path.join(merged_dir, img)
        red, green, red_edge, _ = get_channels(path)

        composite = np.concatenate((green, red_edge, red), axis=-1) * 255
        comp_norm = ((composite - composite.min()) *
                     255 / (composite.max() - composite.min() + 1e-8)).astype(np.uint8)

        out_name = f"processed_{os.path.splitext(img)[0]}.png"
        Image.fromarray(comp_norm).save(os.path.join(out_rgb, out_name))

    print(f"✔ RGB generated for {scene_name}")


# ------------------------------------------------------------
# Step 3 — Extract R/G/RE/NIR PNGs
# ------------------------------------------------------------

def extract_channels(scene_path, merged_dir):
    out_base = scene_path

    folders = ["images_R", "images_G", "images_RE", "images_NIR"]
    for f in folders:
        ensure(os.path.join(out_base, f))

    imgs = [f for f in os.listdir(merged_dir) if f.lower().endswith(('.tif', '.tiff'))]

    for img in imgs:
        path = os.path.join(merged_dir, img)
        red, green, red_edge, nir = get_channels(path)

        base = os.path.splitext(img)[0]

        Image.fromarray(normalize(red)).save(os.path.join(out_base, "images_R",  f"{base}_red.png"))
        Image.fromarray(normalize(green)).save(os.path.join(out_base, "images_G",  f"{base}_green.png"))
        Image.fromarray(normalize(red_edge)).save(os.path.join(out_base, "images_RE", f"{base}_red_edge.png"))
        Image.fromarray(normalize(nir)).save(os.path.join(out_base, "images_NIR", f"{base}_NIR.png"))

    print(f"✔ Channels extracted for {os.path.basename(scene_path)}")


# ------------------------------------------------------------
# Batch Runner
# ------------------------------------------------------------

SCENES = ["cotton", "grape", "houseplant", "sorghum", "tomato"]

def run_all(data_dir):
    for scene in SCENES:
        scene_path = os.path.join(data_dir, scene)
        print("\n====================================")
        print(f"Processing SCENE: {scene}")
        print("====================================")

        merged_dir = merge_tiffs(scene_path, scene)
        generate_rgb(scene_path, scene, merged_dir)
        extract_channels(scene_path, merged_dir)

        print(f"✔ Done with: {scene}")


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to dataset directory containing all scenes")
    args = parser.parse_args()
    run_all(args.data_dir)

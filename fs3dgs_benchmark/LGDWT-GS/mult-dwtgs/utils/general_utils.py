#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import sys
from datetime import datetime
import numpy as np
import random

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def load_multispectral_image(image, resolution, num_bands=4):
    """
    Load multi-spectral image with specified number of bands.
    Assumes the image is stored as a multi-channel image (e.g., 4-channel PNG).
    
    Args:
        image: PIL Image object or path to the image file
        resolution: Target resolution (width, height)
        num_bands: Number of spectral bands (default: 4 for Red, Green, RedEdge, NIR)
    
    Returns:
        torch.Tensor: Image tensor of shape (num_bands, height, width)
    """
    from PIL import Image
    import numpy as np
    
    # Handle both PIL Image objects and file paths
    if isinstance(image, str):
        # If it's a file path, open the image
        pil_image = Image.open(image)
    else:
        # If it's already a PIL Image object, use it directly
        pil_image = image
    
    # Ensure the image is fully loaded
    try:
        pil_image.load()
    except:
        pass
    
    # Convert to numpy array first, then resize
    image_array = np.array(pil_image)
    
    # Handle different image formats
    if len(image_array.shape) == 2:  # Grayscale
        # Repeat grayscale channel for all bands
        image_array = np.stack([image_array] * num_bands, axis=-1)
    elif len(image_array.shape) == 3:
        if image_array.shape[2] == num_bands:
            # Already has the right number of channels
            pass
        elif image_array.shape[2] == 3:  # RGB
            # For RGB images, we'll pad with zeros for additional bands
            if num_bands > 3:
                padding = np.zeros((image_array.shape[0], image_array.shape[1], num_bands - 3))
                image_array = np.concatenate([image_array, padding], axis=-1)
        elif image_array.shape[2] > num_bands:
            # Take first num_bands channels
            image_array = image_array[:, :, :num_bands]
        else:
            # Pad with zeros if we have fewer channels than needed
            padding = np.zeros((image_array.shape[0], image_array.shape[1], num_bands - image_array.shape[2]))
            image_array = np.concatenate([image_array, padding], axis=-1)
    
    # Resize if needed
    if image_array.shape[:2] != resolution:
        # Convert back to PIL for resizing
        pil_image = Image.fromarray(image_array.astype(np.uint8))
        resized_image = pil_image.resize(resolution)
        image_array = np.array(resized_image)
    
    # Normalize to [0, 1]
    image_array = image_array.astype(np.float32) / 255.0
    
    # Convert to torch tensor and permute to (C, H, W)
    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
    
    return image_tensor

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def safe_state(silent):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))

def load_nir_image(image_path, resolution):
    """
    Load a NIR image and resize it to the specified resolution.
    
    Args:
        image_path: Path to the NIR image file
        resolution: Target resolution (width, height)
    
    Returns:
        torch.Tensor: NIR image tensor of shape (1, height, width) normalized to [0, 1]
    """
    from PIL import Image
    import numpy as np
    
    if isinstance(image_path, str):
        pil_image = Image.open(image_path)
    else:
        pil_image = image_path
    
    # Ensure the image is fully loaded
    try:
        pil_image.load()
    except:
        pass
    
    # Convert to grayscale if needed
    if pil_image.mode != 'L':
        pil_image = pil_image.convert('L')
    
    # Resize the image
    resized_image = pil_image.resize(resolution)
    
    # Convert to numpy array
    image_array = np.array(resized_image)
    
    # Normalize to [0, 1] (keep linear, no gamma correction)
    image_array = image_array.astype(np.float32) / 255.0
    
    # Convert to tensor and add channel dimension (1, height, width)
    tensor = torch.from_numpy(image_array).unsqueeze(0)
    
    return tensor

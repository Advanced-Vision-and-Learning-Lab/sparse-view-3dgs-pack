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
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
try:
    from diff_gaussian_rasterization._C import fusedssim, fusedssim_backward
except:
    pass

C1 = 0.01 ** 2
C2 = 0.03 ** 2

class FusedSSIMMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C1, C2, img1, img2):
        ssim_map = fusedssim(C1, C2, img1, img2)
        ctx.save_for_backward(img1.detach(), img2)
        ctx.C1 = C1
        ctx.C2 = C2
        return ssim_map

    @staticmethod
    def backward(ctx, opt_grad):
        img1, img2 = ctx.saved_tensors
        C1, C2 = ctx.C1, ctx.C2
        grad = fusedssim_backward(C1, C2, img1, img2, opt_grad)
        return None, None, grad, None

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def fast_ssim(img1, img2):
    ssim_map = FusedSSIMMap.apply(C1, C2, img1, img2)
    return ssim_map.mean()


# ------------------------------
# DWT loss utilities using pytorch-wavelets
# ------------------------------

def charbonnier_loss(pred: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-3) -> torch.Tensor:
    """Charbonnier loss over all dimensions."""
    diff = pred - target
    return torch.sqrt(diff * diff + (epsilon * epsilon)).mean()


from pytorch_wavelets import DWTForward

def get_dwt_subbands(x: torch.Tensor) -> dict:
    """Get all DWT subbands using pytorch_wavelets package.
    
    Args:
        x: (N, C, H, W) input tensor
        
    Returns:
        Dictionary with keys: 'LL1', 'LH1', 'HL1', 'HH1', 'LL2', 'LH2', 'HL2', 'HH2'
    """
    device = x.device
    dtype = x.dtype
    
    # Initialize DWT for 2 levels using Haar (db1) wavelet
    # J=2 means it computes Level 1 and Level 2
    # mode='symmetric' is similar to reflect padding
    dwt = DWTForward(J=2, mode='symmetric', wave='db1').to(device)
    
    # Yl is the low-pass coefficients at the coarsest level (LL2)
    # Yh is a list of high-pass coefficients at each level (fine to coarse)
    # Yh[0] contains (LH1, HL1, HH1)
    # Yh[1] contains (LH2, HL2, HH2)
    Yl, Yh = dwt(x)
    
    LL2 = Yl
    
    # Level 1 high-pass
    LH1, HL1, HH1 = Yh[0][:,:,0,:,:], Yh[0][:,:,1,:,:], Yh[0][:,:,2,:,:]
    
    # Level 2 high-pass
    LH2, HL2, HH2 = Yh[1][:,:,0,:,:], Yh[1][:,:,1,:,:], Yh[1][:,:,2,:,:]
    
    # To get LL1, we can run 1-level DWT 

    
    dwt1 = DWTForward(J=1, mode='symmetric', wave='db1').to(device)
    
    # Level 1
    LL1, Yh1 = dwt1(x)
    LH1, HL1, HH1 = Yh1[0][:,:,0,:,:], Yh1[0][:,:,1,:,:], Yh1[0][:,:,2,:,:]
    
    # Level 2 (input is LL1)
    LL2, Yh2 = dwt1(LL1)
    LH2, HL2, HH2 = Yh2[0][:,:,0,:,:], Yh2[0][:,:,1,:,:], Yh2[0][:,:,2,:,:]
    
    return {
        'LL1': LL1, 'LH1': LH1, 'HL1': HL1, 'HH1': HH1,
        'LL2': LL2, 'LH2': LH2, 'HL2': HL2, 'HH2': HH2,
    }


# ------------------------------
# Wavelet Error Field (WEF)
# ------------------------------

def _normalize_to_01(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    min_v = x.amin(dim=(-2, -1), keepdim=True)
    max_v = x.amax(dim=(-2, -1), keepdim=True)
    return (x - min_v) / (max_v - min_v + eps)

def compute_wef_maps(pred: torch.Tensor, gt: torch.Tensor) -> dict:
    """Compute Wavelet Error Field (WEF) maps for 2-level Haar DWT on residual.

    Args:
        pred: (N,C,H,W) predicted image in [0,1]
        gt:   (N,C,H,W) ground truth image in [0,1]

    Returns:
        Dict with HxW upsampled heatmaps per band and a combined map:
        {'LL2': ..., 'LH2': ..., 'HL2': ..., 'WEF': ...}
    """
    assert pred.shape == gt.shape, "pred and gt must have same shape"
    device = pred.device

    residual = pred - gt  # signed residual
    bands = get_dwt_subbands(residual)

    # Take level-2 bands
    LL2 = bands['LL2']  # (N,C,h,w)
    LH2 = bands['LH2']
    HL2 = bands['HL2']

    # Energy maps per channel
    e_LL2 = LL2 * LL2
    e_LH2 = LH2 * LH2
    e_HL2 = HL2 * HL2

    # Energy normalization : LL2×4, LH2×2, HL2×2
    e_LL2 = e_LL2 * 4.0
    e_LH2 = e_LH2 * 2.0
    e_HL2 = e_HL2 * 2.0

    # Aggregate across channels
    e_LL2 = e_LL2.mean(dim=1, keepdim=True)
    e_LH2 = e_LH2.mean(dim=1, keepdim=True)
    e_HL2 = e_HL2.mean(dim=1, keepdim=True)

    # Upsample to input resolution
    H, W = pred.shape[-2:]
    up_LL2 = F.interpolate(e_LL2, size=(H, W), mode='bilinear', align_corners=False)
    up_LH2 = F.interpolate(e_LH2, size=(H, W), mode='bilinear', align_corners=False)
    up_HL2 = F.interpolate(e_HL2, size=(H, W), mode='bilinear', align_corners=False)

    # Normalize each map to [0,1] for visualization
    n_LL2 = _normalize_to_01(up_LL2)
    n_LH2 = _normalize_to_01(up_LH2)
    n_HL2 = _normalize_to_01(up_HL2)

    # Combined WEF as average of normalized maps
    wef = (n_LL2 + n_LH2 + n_HL2) / 3.0

    return {
        'LL2': n_LL2,  # (N,1,H,W)
        'LH2': n_LH2,
        'HL2': n_HL2,
        'WEF': _normalize_to_01(wef)
    }

def make_heatmap_rgb(x01: torch.Tensor) -> torch.Tensor:
    """Convert single-channel [0,1] map to 3-channel pseudo-color heatmap (simple jet-like).
    Args:
        x01: (N,1,H,W)
    Returns:
        (N,3,H,W) heatmap in [0,1]
    """
    # Simple 3-color mapping: blue->green->red
    x = x01.clamp(0, 1)
    r = x
    g = 1.0 - (x - 0.5).abs() * 2.0
    b = 1.0 - x
    g = g.clamp(0, 1)
    return torch.cat([r, g, b], dim=1)


def make_wef_grid_image(maps: dict, keys: list, tile_cols: int = 3) -> 'Image.Image':
    """Create a PIL grid image from heatmap keys using make_heatmap_rgb.
    Expects maps[key] tensors in (N,1,H,W), uses the first sample.
    """
    from PIL import Image as _PILImage
    imgs = []
    for k in keys:
        hm = make_heatmap_rgb(maps[k])[0]  # (3,H,W)
        arr = (hm.permute(1,2,0).detach().cpu().clamp(0,1).numpy() * 255).astype('uint8')
        imgs.append(_PILImage.fromarray(arr))
    if not imgs:
        raise ValueError("No images for grid")
    w, h = imgs[0].size
    cols = tile_cols
    rows = (len(imgs) + cols - 1) // cols
    grid = _PILImage.new('RGB', (cols * w, rows * h))
    for idx, im in enumerate(imgs):
        r = idx // cols
        c = idx % cols
        grid.paste(im, (c * w, r * h))
    return grid


def make_wef_grid_image_titled(maps: dict, keys: list, tile_cols: int = 3) -> 'Image.Image':
    """Create a grid like make_wef_grid_image, but draw band titles on each tile."""
    from PIL import Image as _PILImage, ImageDraw as _Draw, ImageFont as _Font
    base = make_wef_grid_image(maps, keys, tile_cols)
    draw = _Draw.Draw(base)
    try:
        font = _Font.load_default()
    except Exception:
        font = None
    # tile size
    hm = make_heatmap_rgb(maps[keys[0]])[0]
    w, h = hm.permute(1,2,0).shape[1], hm.permute(1,2,0).shape[0]
    cols = tile_cols
    for idx, k in enumerate(keys):
        r = idx // cols
        c = idx % cols
        x = c * w + 6
        y = r * h + 6
        draw.rectangle([c*w, r*h, c*w+120, r*h+26], fill=(0,0,0,128))
        draw.text((x, y), k, fill=(255,255,255), font=font)
    return base

def compute_wef_all_subbands(pred: torch.Tensor, gt: torch.Tensor) -> dict:
    """Compute WEF maps for all 8 subbands + combined.

    Normalization factors (example policy):
      - Level 1: LL1×1, LH1×1, HL1×1, HH1×1
      - Level 2: LL2×4, LH2×2, HL2×2, HH2×2

    Returns maps normalized to [0,1], upsampled to (H,W).
    Keys: LL1,LH1,HL1,HH1,LL2,LH2,HL2,HH2,COMBINED
    """
    assert pred.shape == gt.shape
    H, W = pred.shape[-2:]
    residual = pred - gt
    b = get_dwt_subbands(residual)

    # Energy per band
    energies = {
        'LL1': b['LL1'] * b['LL1'],
        'LH1': b['LH1'] * b['LH1'],
        'HL1': b['HL1'] * b['HL1'],
        'HH1': b['HH1'] * b['HH1'],
        'LL2': b['LL2'] * b['LL2'],
        'LH2': b['LH2'] * b['LH2'],
        'HL2': b['HL2'] * b['HL2'],
        'HH2': b['HH2'] * b['HH2'],
    }

    # Scale factors
    scale = {
        'LL1': 1.0, 'LH1': 1.0, 'HL1': 1.0, 'HH1': 1.0,
        'LL2': 4.0, 'LH2': 2.0, 'HL2': 2.0, 'HH2': 2.0,
    }

    maps = {}
    for k, e in energies.items():
        e = e * scale[k]
        e = e.mean(dim=1, keepdim=True)  # across channels
        e = F.interpolate(e, size=(H, W), mode='bilinear', align_corners=False)
        maps[k] = _normalize_to_01(e)

    # Combined as average of all normalized maps
    combo = None
    for k in ['LL1','LH1','HL1','HH1','LL2','LH2','HL2','HH2']:
        combo = maps[k] if combo is None else (combo + maps[k])
    maps['COMBINED'] = _normalize_to_01(combo / 8.0)
    return maps


# ------------------------------
# ELF and Patch-wise DWT Utilities
# ------------------------------

def compute_elf_map(image: torch.Tensor) -> torch.Tensor:
    """Compute Energy of Low Frequency (ELF) map per pixel.
    ELF(x, y) = ||LL(x, y)||1 / (||LL(x, y)||1 + ||HF(x, y)||1)
    
    Args:
        image: (N, C, H, W) input image
    
    Returns:
        elf_map: (N, 1, H, W) ELF scores in [0, 1], upsampled to input size.
    """
    bands = get_dwt_subbands(image)
    
    # L1 norm across channels
    def l1(x): return torch.sum(torch.abs(x), dim=1, keepdim=True)
    
    LL = l1(bands['LL1'])
    LH = l1(bands['LH1'])
    HL = l1(bands['HL1'])
    HH = l1(bands['HH1'])
    
    HF = LH + HL + HH
    
    # Compute ELF at 1/2 resolution
    # Add epsilon to denominator to avoid division by zero
    elf_low = LL / (LL + HF + 1e-8)
    
    # Upsample to full resolution
    H, W = image.shape[-2:]
    elf_map = F.interpolate(elf_low, size=(H, W), mode='bilinear', align_corners=False)
    
    return elf_map

def compute_patch_dwt_loss(pred: torch.Tensor, gt: torch.Tensor, elf_map: torch.Tensor, 
                          patch_size: int = 128, percentile: float = 0.2,
                          lh1_weight: float = 1.0, hl1_weight: float = 1.0) -> torch.Tensor:
    """Compute Patch-wise DWT loss on regions with HIGH ELF (smooth areas).
    
    Args:
        pred: (N, C, H, W) rendered image
        gt: (N, C, H, W) ground truth image
        elf_map: (N, 1, H, W) ELF map computed from GT
        patch_size: size of patches (default 128)
        percentile: threshold for selecting HIGH-ELF patches (default 0.2 = top 20%)
        lh1_weight: weight for LH1 subband loss
        hl1_weight: weight for HL1 subband loss
        
    Returns:
        loss: scalar tensor
    """
    N, C, H, W = pred.shape
    
    # Unfold images into non-overlapping patches
    stride = patch_size
    
    if H < patch_size or W < patch_size:
        return torch.tensor(0.0, device=pred.device)

    # Use unfold to extract patches
    pred_patches = F.unfold(pred, kernel_size=patch_size, stride=stride)
    gt_patches = F.unfold(gt, kernel_size=patch_size, stride=stride)
    elf_patches = F.unfold(elf_map, kernel_size=patch_size, stride=stride)
    
    # Reshape to (N, L, C, H, W) roughly
    L = pred_patches.shape[2]
    
    # Compute mean ELF per patch
    patch_elf_means = elf_patches.mean(dim=1) # (N, L)
    
    # Determine threshold for HIGH ELF (smooth areas)
    # We want top 'percentile' patches (e.g. top 20%)
    all_elf_means = patch_elf_means.view(-1)
    
    # To get top 20%, we need the value at (1 - percentile) quantile
    # e.g. if percentile=0.2, we want top 20%, so we look for 80th percentile value
    k = int(all_elf_means.numel() * (1.0 - percentile))
    k = max(1, k)
    k = min(k, all_elf_means.numel())
    
    threshold, _ = torch.kthvalue(all_elf_means, k)
    
    # Select patches with HIGH ELF (Smooth areas)
    mask = patch_elf_means >= threshold # (N, L)
    
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device)
    
    # Extract selected patches
    pred_patches = pred_patches.view(N, C, patch_size, patch_size, L).permute(0, 4, 1, 2, 3)
    gt_patches = gt_patches.view(N, C, patch_size, patch_size, L).permute(0, 4, 1, 2, 3)
    
    # Flatten batch and patches
    pred_sel = pred_patches[mask] # (N_sel, C, ps, ps)
    gt_sel = gt_patches[mask]     # (N_sel, C, ps, ps)
    
    # Apply DWT to selected patches
    pred_bands = get_dwt_subbands(pred_sel)
    gt_bands = get_dwt_subbands(gt_sel)
    
    # Loss on HF bands (LH, HL, HH) - penalize high frequency in smooth areas
    loss_LH = l1_loss(pred_bands['LH1'], gt_bands['LH1'])
    loss_HL = l1_loss(pred_bands['HL1'], gt_bands['HL1'])
    loss_HH = l1_loss(pred_bands['HH1'], gt_bands['HH1'])

    
    loss = (lh1_weight * loss_LH) + (hl1_weight * loss_HL) + (0.5 * (lh1_weight + hl1_weight) * loss_HH)
    
    return loss

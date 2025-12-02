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


# class DWTLossScaler:
#     """Automatic scaling for DWT losses to match base loss magnitude."""
    
#     def __init__(self, target_scale: float = 1.0, momentum: float = 0.9):
#         self.target_scale = target_scale
#         self.momentum = momentum
#         self.dwt_scale = 1.0
#         self.base_loss_ema = None
#         self.dwt_loss_ema = None
#         self.initialized = False
        
#     def update_scales(self, base_loss: torch.Tensor, dwt_loss: torch.Tensor):
#         """Update scaling factors based on loss magnitudes."""
#         base_mag = base_loss.item()
#         dwt_mag = dwt_loss.item()
        
#         # Initialize EMAs
#         if not self.initialized:
#             self.base_loss_ema = base_mag
#             self.dwt_loss_ema = dwt_mag
#             self.initialized = True
#         else:
#             # Update EMAs
#             self.base_loss_ema = self.momentum * self.base_loss_ema + (1 - self.momentum) * base_mag
#             self.dwt_loss_ema = self.momentum * self.dwt_loss_ema + (1 - self.momentum) * dwt_mag
        
#         # Calculate scale factor to match target scale
#         # IMPORTANT: Don't compensate for user's dwt_weight setting
#         if self.dwt_loss_ema > 1e-8:  # Avoid division by zero
#             # Target is to make DWT loss magnitude similar to base loss
#             # But don't interfere with user's weight multiplier
#             target_dwt_mag = self.base_loss_ema * self.target_scale
#             self.dwt_scale = target_dwt_mag / self.dwt_loss_ema
#         else:
#             self.dwt_scale = 1.0
    
#     def scale_dwt_loss(self, dwt_loss: torch.Tensor) -> torch.Tensor:
#         """Apply scaling to DWT loss."""
#         return dwt_loss * self.dwt_scale
    
#     def get_scale_info(self) -> dict:
#         """Get current scaling information."""
#         return {
#             'dwt_scale': self.dwt_scale,
#             'base_loss_ema': self.base_loss_ema,
#             'dwt_loss_ema': self.dwt_loss_ema,
#             'target_scale': self.target_scale
#         }


def get_dwt_subbands(x: torch.Tensor) -> dict:
    """Get all DWT subbands using fast GPU implementation.
    
    Args:
        x: (N, C, H, W) input tensor
        
    Returns:
        Dictionary with keys: 'LL1', 'LH1', 'HL1', 'HH1', 'LL2', 'LH2', 'HL2', 'HH2'
    """
    device = x.device
    dtype = x.dtype
    
    # Haar wavelet filters
    inv_sqrt2 = 1.0 / (2.0 ** 0.5)
    h = torch.tensor([inv_sqrt2, inv_sqrt2], device=device, dtype=dtype).view(1, 1, 2)
    g = torch.tensor([inv_sqrt2, -inv_sqrt2], device=device, dtype=dtype).view(1, 1, 2)
    
    # Construct 2D kernels via outer products
    hh = (h.transpose(-1, -2) @ h).view(1, 1, 2, 2)  # LL
    gh = (g.transpose(-1, -2) @ h).view(1, 1, 2, 2)  # LH
    hg = (h.transpose(-1, -2) @ g).view(1, 1, 2, 2)  # HL
    gg = (g.transpose(-1, -2) @ g).view(1, 1, 2, 2)  # HH
    
    def dwt_level(x):
        """Apply one level of DWT - fast GPU implementation."""
        n, c, h, w = x.shape
        
        # Expand kernels for depthwise convolution
        hh_exp = hh.expand(c, 1, 2, 2)
        gh_exp = gh.expand(c, 1, 2, 2)
        hg_exp = hg.expand(c, 1, 2, 2)
        gg_exp = gg.expand(c, 1, 2, 2)
        
        # Apply reflect padding and convolution
        x_pad = F.pad(x, (1, 1, 1, 1), mode='reflect')
        
        LL = F.conv2d(x_pad, hh_exp, stride=2, groups=c)
        LH = F.conv2d(x_pad, gh_exp, stride=2, groups=c)
        HL = F.conv2d(x_pad, hg_exp, stride=2, groups=c)
        HH = F.conv2d(x_pad, gg_exp, stride=2, groups=c)
        
        return LL, LH, HL, HH
    
    # Level 1 DWT
    LL1, LH1, HL1, HH1 = dwt_level(x)
    
    # Level 2 DWT (on LL1)
    LL2, LH2, HL2, HH2 = dwt_level(LL1)
    
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





# # ------------------------------
# # Selection maps (LL~, M, S, HH1~, veto) on RGB residuals
# # ------------------------------

# def _percentile95(x: torch.Tensor) -> torch.Tensor:
#     """Compute 95th percentile per sample for (N,1,H,W) → (N,1,1,1)."""
#     x_flat = x.view(x.shape[0], -1)
#     # torch.quantile is available in recent PyTorch; fallback to topk if missing
#     try:
#         p = torch.quantile(x_flat, 0.95, dim=1, keepdim=True)
#     except Exception:
#         k = (x_flat.shape[1] * 95) // 100
#         k = max(1, min(k, x_flat.shape[1]))
#         topk, _ = torch.topk(x_flat, k, dim=1, largest=True)
#         p = topk[:, -1:]
#     p = p.view(-1, 1, 1, 1)
#     return p.clamp(min=1e-6)


def build_selection_maps(pred: torch.Tensor, gt: torch.Tensor, use_blur_ll2: bool = True,
                         a: float = 1.0, b: float = 0.3) -> dict:
    """Build selection maps using the working approach from CVCF tests.
    
    - Compute DWT subbands of pred and gt separately
    - Compute error in each subband: |pred_band - gt_band|
    - Normalize to 95th percentile
    - M = a * LL_t + b * (LH_t + HL_t)
    - S = M (use raw error values for better candidate selection)
    - veto = clamp(HH1_t - 0.5, 0, 1)
    """
    assert pred.shape == gt.shape
    
    # Get DWT subbands for pred and gt separately
    pred_bands = get_dwt_subbands(pred)
    gt_bands = get_dwt_subbands(gt)
    
    # Compute error in each subband
    LL_t = torch.abs(pred_bands['LL1'] - gt_bands['LL1'])
    LH_t = torch.abs(pred_bands['LH1'] - gt_bands['LH1'])
    HL_t = torch.abs(pred_bands['HL1'] - gt_bands['HL1'])
    HH1_t = torch.abs(pred_bands['HH1'] - gt_bands['HH1'])
    
    # Normalize to 95th percentile
    def normalize_95th(x):
        flat = x.view(x.shape[0], -1)
        p95 = torch.quantile(flat, 0.95, dim=1, keepdim=True)
        p95 = p95.view(x.shape[0], 1, 1, 1)
        return x / (p95 + 1e-8)
    
    LL_t = normalize_95th(LL_t)
    LH_t = normalize_95th(LH_t)
    HL_t = normalize_95th(HL_t)
    HH1_t = normalize_95th(HH1_t)
    

    # Compute M (multi-scale error)
    M = a * LL_t + b * (LH_t + HL_t)
    
    # Use raw error values for better candidate selection (don't normalize to [0,1])
    S = M
    
    return {
        'LL_t': LL_t,
        'LH_t': LH_t,
        'HL_t': HL_t,
        'HH1_t': HH1_t,
        'M': M,
        'S': S
    }


def build_selection_maps_simple(pred: torch.Tensor, gt: torch.Tensor,
                                subbands: tuple = ('LL2', 'LH2', 'HL2'),
                                band_scales: dict = None,
                                a: float = 1.0, b: float = 0.3) -> dict:
    """Simplified selection maps using only selected subbands, no blur, no veto.

    Steps:
      - Residual r = pred - gt
      - DWT → selected bands
      - Energy per band (squared), apply per-band scale
      - Channel-mean, upsample to input size
      - M = a*LL + b*(LH+HL) if those bands are present among selection; otherwise sum available
      - S = M (no extra normalization)
    """
    assert pred.shape == gt.shape
    N, C, H, W = pred.shape
    residual = pred - gt
    b = get_dwt_subbands(residual)

    if band_scales is None:
        band_scales = {'LL2': 4.0, 'LH2': 2.0, 'HL2': 2.0}

    # Prepare per-band maps at input res
    def band_energy_up(k):
        e = b[k] * b[k]
        e = e * band_scales.get(k, 1.0)
        e = e.mean(dim=1, keepdim=True)
        return F.interpolate(e, size=(H, W), mode='bilinear', align_corners=False)

    maps = {}
    for k in subbands:
        if k not in b:
            raise ValueError(f"Unknown subband '{k}'")
        maps[k] = band_energy_up(k)

    # Build M using available components
    LL_term = maps['LL2'] if 'LL2' in maps else 0.0
    LH_term = maps['LH2'] if 'LH2' in maps else 0.0
    HL_term = maps['HL2'] if 'HL2' in maps else 0.0
    if isinstance(LL_term, float):
        # No LL2 present; sum all available maps equally
        acc = None
        for v in maps.values():
            acc = v if acc is None else (acc + v)
        M = acc / max(1, len(maps))
    else:
        M = a * LL_term + b * (LH_term + HL_term)

    S = M  # raw

    return {
        'S': S,
        'M': M,
        **{k: maps[k] for k in maps}
    }



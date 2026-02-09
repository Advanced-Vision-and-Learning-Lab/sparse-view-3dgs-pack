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

def create_window(window_size, channel, dtype=torch.float32):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).to(dtype).unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel, dtype=img1.dtype)

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

def l1_loss_nir(pred, target):
    """
    L1 loss for NIR images (single channel).
    
    Args:
        pred: Predicted NIR image (1, H, W)
        target: Ground truth NIR image (1, H, W)
    
    Returns:
        torch.Tensor: L1 loss value
    """
    return F.l1_loss(pred, target)

def ssim_loss_nir(pred, target):
    """
    SSIM loss for NIR images (single channel).
    
    Args:
        pred: Predicted NIR image (1, H, W)
        target: Ground truth NIR image (1, H, W)
    
    Returns:
        torch.Tensor: SSIM loss value (1 - SSIM)
    """
    # Ensure both images have the same dtype
    pred = pred.to(target.dtype)
    
    # For single channel, we can use the existing SSIM function
    # by treating it as a 3-channel image with repeated channels
    pred_3ch = pred.repeat(3, 1, 1).unsqueeze(0)  # (1, 3, H, W)
    target_3ch = target.repeat(3, 1, 1).unsqueeze(0)  # (1, 3, H, W)
    
    ssim_value = ssim(pred_3ch, target_3ch)
    return 1.0 - ssim_value

def combined_nir_loss(pred, target, l1_weight=1.0, ssim_weight=0.2):
    """
    Combined L1 + SSIM loss for NIR images.
    
    Args:
        pred: Predicted NIR image (1, H, W)
        target: Ground truth NIR image (1, H, W)
        l1_weight: Weight for L1 loss
        ssim_weight: Weight for SSIM loss
    
    Returns:
        torch.Tensor: Combined loss value
    """
    l1 = l1_loss_nir(pred, target)
    ssim_loss_val = ssim_loss_nir(pred, target)
    
    return l1_weight * l1 + ssim_weight * ssim_loss_val

def compute_combined_residuals(rgb_pred, rgb_gt, nir_pred=None, nir_gt=None):
    """
    Compute combined residual maps for RGB and NIR for enhanced densification.
    
    Args:
        rgb_pred: Predicted RGB image (3, H, W)
        rgb_gt: Ground truth RGB image (3, H, W)
        nir_pred: Predicted NIR image (1, H, W) or None
        nir_gt: Ground truth NIR image (1, H, W) or None
    
    Returns:
        torch.Tensor: Combined residual map (H, W) for densification
    """
    # Compute RGB residuals
    rgb_residual = torch.abs(rgb_pred - rgb_gt).mean(dim=0)  # (H, W)
    
    if nir_pred is not None and nir_gt is not None:
        # Compute NIR residuals
        nir_residual = torch.abs(nir_pred - nir_gt).squeeze(0)  # (H, W)
        
        # Take maximum of RGB and NIR residuals
        combined_residual = torch.max(rgb_residual, nir_residual)
    else:
        # Use only RGB residuals
        combined_residual = rgb_residual
    
    return combined_residual

def compute_dwt_residuals(residual_map):
    """
    Compute Discrete Wavelet Transform residuals for enhanced densification.
    This helps identify fine details like leaf veins and edges.
    
    Args:
        residual_map: Residual map (H, W)
    
    Returns:
        torch.Tensor: DWT high-frequency residuals (H, W)
    """
    # Simple high-pass filter approximation using Sobel-like operations
    # This is a simplified version - in practice, you might want to use actual DWT
    
    # Convert to numpy for easier processing
    residual_np = residual_map.detach().cpu().numpy()
    
    # Simple edge detection (Sobel-like)
    import numpy as np
    
    # Sobel kernels
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    
    # Apply convolution (simplified)
    from scipy import ndimage
    try:
        grad_x = ndimage.convolve(residual_np, sobel_x, mode='constant')
        grad_y = ndimage.convolve(residual_np, sobel_y, mode='constant')
        dwt_residual = np.sqrt(grad_x**2 + grad_y**2)
    except ImportError:
        # Fallback to simple difference if scipy not available
        dwt_residual = np.abs(np.diff(residual_np, axis=0, prepend=residual_np[0:1]))
        dwt_residual = np.abs(np.diff(dwt_residual, axis=1, prepend=dwt_residual[:, 0:1]))
    
    return torch.from_numpy(dwt_residual).to(residual_map.device)

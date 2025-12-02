#!/usr/bin/env python3
"""
Test script to verify pytorch-wavelets integration with real images.
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os
import sys
from utils.loss_utils import get_dwt_subbands, get_dwt_subbands_fallback, charbonnier_loss

def load_image(image_path):
    """Load and preprocess an image."""
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return None
    
    # Load image
    img = Image.open(image_path).convert('RGB')
    print(f"Original image size: {img.size}")
    
    # Convert to tensor and normalize to [0, 1]
    img_array = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # (C, H, W)
    
    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0)  # (1, C, H, W)
    
    return img_tensor

def test_pytorch_wavelets(image_path=None):
    """Test pytorch-wavelets integration."""
    print("Testing pytorch-wavelets integration...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load real image or create synthetic one
    if image_path:
        print(f"Loading image: {image_path}")
        x = load_image(image_path)
        if x is None:
            return False
    else:
        print("No image path provided, creating synthetic test image")
        # Create test image (1, 3, 64, 64)
        x = torch.randn(1, 3, 64, 64, device=device)
        x = torch.clamp(x, 0, 1)
    
    # Move to device
    x = x.to(device)
    
    print(f"Input shape: {x.shape}")
    print(f"Input range: [{x.min():.3f}, {x.max():.3f}]")
    
    try:
        # Try pytorch-wavelets
        bands = get_dwt_subbands(x)
        print("✓ pytorch-wavelets available")
        
        print("\nDWT subband shapes:")
        for name, band in bands.items():
            print(f"  {name}: {band.shape}")
            print(f"    Range: [{band.min():.3f}, {band.max():.3f}]")
        
        # Test Charbonnier loss
        pred = bands['LL1']
        target = bands['LL1'] + 0.1 * torch.randn_like(bands['LL1'])
        
        loss = charbonnier_loss(pred, target)
        print(f"\nCharbonnier loss: {loss.item():.6f}")
        
        # Test with different subbands
        print("\nTesting different subband losses:")
        for name in ['LL1', 'LL2', 'LH1', 'HL1', 'HH1']:
            if name in bands:
                pred_band = bands[name]
                target_band = bands[name] + 0.05 * torch.randn_like(bands[name])
                band_loss = charbonnier_loss(pred_band, target_band)
                print(f"  {name} loss: {band_loss.item():.6f}")
        
        return True
        
    except ImportError:
        print("✗ pytorch-wavelets not available, testing fallback")
        
        # Test fallback
        bands = get_dwt_subbands_fallback(x)
        
        print("\nFallback DWT subband shapes:")
        for name, band in bands.items():
            print(f"  {name}: {band.shape}")
            print(f"    Range: [{band.min():.3f}, {band.max():.3f}]")
        
        return False

def main():
    """Run pytorch-wavelets test."""
    print("=" * 50)
    print("PyTorch-Wavelets Integration Test")
    print("=" * 50)
    
    # Check if image path provided as command line argument
    image_path = None
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        print(f"Testing with image: {image_path}")
    else:
        print("No image path provided, using synthetic data")
        print("Usage: python test_pytorch_wavelets.py <image_path>")
    
    success = test_pytorch_wavelets(image_path)
    
    if success:
        print("\n" + "=" * 50)
        print("✓ pytorch-wavelets integration successful!")
        print("=" * 50)
    else:
        print("\n" + "=" * 50)
        print("⚠ pytorch-wavelets not available, using fallback")
        print("Install with: pip install pytorch-wavelets")
        print("=" * 50)
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

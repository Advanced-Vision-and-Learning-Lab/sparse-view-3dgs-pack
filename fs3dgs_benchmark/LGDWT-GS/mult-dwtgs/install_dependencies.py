#!/usr/bin/env python3
"""
Install missing dependencies for multi-spectral 3DGS.
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a package using pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package}")
        return False

def check_and_install_dependencies():
    """Check and install required dependencies."""
    print("🔧 Checking and installing dependencies for Multi-spectral 3DGS...\n")
    
    # List of required packages
    required_packages = [
        "plyfile",
        "tqdm", 
        "torch",
        "torchvision",
        "numpy",
        "Pillow",
        "opencv-python",
        "tensorboard"
    ]
    
    missing_packages = []
    
    # Check which packages are missing
    for package in required_packages:
        try:
            if package == "opencv-python":
                import cv2
            elif package == "Pillow":
                import PIL
            else:
                __import__(package)
            print(f"✅ {package} is already installed")
        except ImportError:
            print(f"❌ {package} is missing")
            missing_packages.append(package)
    
    # Install missing packages
    if missing_packages:
        print(f"\n📦 Installing {len(missing_packages)} missing packages...")
        for package in missing_packages:
            install_package(package)
    else:
        print("\n🎉 All required packages are already installed!")
    
    print("\n📋 Additional 3DGS-specific dependencies:")
    print("   For full 3DGS functionality, you may also need:")
    print("   - diff-gaussian-rasterization (custom CUDA extension)")
    print("   - simple-knn (custom CUDA extension)")
    print("   - fused-ssim (optional, for faster SSIM computation)")
    print("\n   These are typically installed from the original 3DGS repository.")
    
    return len(missing_packages) == 0

if __name__ == "__main__":
    success = check_and_install_dependencies()
    
    if success:
        print("\n🚀 Dependencies check complete!")
        print("You can now run the multi-spectral training script.")
    else:
        print("\n⚠️  Some dependencies failed to install.")
        print("Please check the error messages above and install manually if needed.")

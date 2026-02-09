# Multi-Spectral 3D Gaussian Splatting

This is an extension of 3D Gaussian Splatting (3DGS) that supports multi-spectral rendering with 4-band data (Red, Green, RedEdge, NIR). The implementation adds a learnable spectral head that converts RGB renderings to multi-spectral predictions.

## Overview

The multi-spectral 3DGS works by:

1. **Standard 3DGS Rendering**: First, the standard 3D Gaussian Splatting renders an RGB image as usual
2. **Spectral Head**: A small learnable neural network (spectral head) takes the RGB rendering and predicts 4 spectral bands
3. **Joint Training**: Both the 3D Gaussians and the spectral head are trained jointly using L1 loss on the spectral predictions

## Key Components

### Spectral Head
- **SimpleSpectralHead**: A lightweight linear transformation that maps RGB pixels to 4-band spectral data
- **SpectralHead**: A more complex CNN-based head (alternative implementation)
- The spectral head learns to translate RGB appearance to spectral responses

### Multi-Spectral Data Loading
- Supports 4-channel images (Red, Green, RedEdge, NIR)
- Images should be stored in the same format as RGB images but with 4 channels
- Compatible with standard COLMAP dataset structure

## Usage

### 1. Dataset Preparation

Your dataset should follow the standard 3DGS structure:

```
your_dataset/
├── images/           # 4-channel images (Red, Green, RedEdge, NIR)
│   ├── image1.png
│   ├── image2.png
│   └── ...
├── sparse/           # COLMAP reconstruction
│   ├── cameras.bin
│   ├── images.bin
│   └── points3D.bin
└── ...
```

**Important**: The images in the `images/` folder should be 4-channel images where:
- Channel 0: Red
- Channel 1: Green  
- Channel 2: RedEdge
- Channel 3: NIR

### 2. Training

#### Using the example script:
```bash
python example_multispectral.py --source_path /path/to/your/dataset --use_multispectral
```

#### Using the standard training script:
```bash
python train.py -s /path/to/your/dataset --use_multispectral --num_spectral_bands 4
```

### 3. Rendering

After training, render multi-spectral images:
```bash
python render.py -m /path/to/trained/model --use_multispectral
```

## Key Parameters

- `--use_multispectral`: Enable multi-spectral training
- `--num_spectral_bands`: Number of spectral bands (default: 4)
- `--spectral_lr`: Learning rate for spectral head (default: 0.001)

## Architecture Details

### Spectral Head Implementation

The `SimpleSpectralHead` uses a per-pixel linear transformation:

```python
# For each pixel (R, G, B) -> (Red, Green, RedEdge, NIR)
spectral = linear_transform(rgb_pixel)
```

This is implemented as a single linear layer that maps 3 RGB values to 4 spectral values.

### Training Process

1. **RGB Rendering**: Standard 3DGS renders RGB image
2. **Spectral Prediction**: Spectral head converts RGB to 4-band spectral image
3. **Loss Computation**: L1 loss between predicted and ground-truth spectral images
4. **Joint Optimization**: Both 3D Gaussians and spectral head are updated

### Loss Function

The total loss combines:
- Standard RGB loss (L1 + SSIM)
- Spectral loss (L1 on 4-band predictions)

```python
total_loss = rgb_loss + spectral_loss
```

## File Structure

Key files added/modified:

- `spectral_head.py`: Spectral head implementations
- `scene/gaussian_model.py`: Modified to include spectral head
- `gaussian_renderer/__init__.py`: Modified renderer to output spectral images
- `scene/cameras.py`: Modified to load multi-spectral images
- `utils/general_utils.py`: Added multi-spectral image loading
- `train.py`: Modified training loop for spectral loss
- `render.py`: Modified to save spectral images
- `arguments/__init__.py`: Added multi-spectral parameters

## Example Output

After training, you'll get:
- Standard RGB renderings
- 4-band spectral renderings (Red, Green, RedEdge, NIR)
- Ground truth spectral images for comparison

The spectral images are saved separately for each band:
- `00000_spectral_Red.png`
- `00000_spectral_Green.png`
- `00000_spectral_RedEdge.png`
- `00000_spectral_NIR.png`

## Tips for Best Results

1. **Data Quality**: Ensure your 4-band images are properly calibrated and aligned
2. **Training Time**: Multi-spectral training may take slightly longer due to the additional spectral head
3. **Learning Rate**: The spectral head learning rate (0.001) works well in most cases
4. **Loss Weighting**: Currently, spectral loss has equal weight to RGB loss; you can modify this in the training code

## Compatibility

This extension is fully compatible with:
- Standard 3DGS datasets (RGB mode)
- COLMAP reconstructions
- All existing 3DGS features (densification, pruning, etc.)

## Troubleshooting

1. **Import Errors**: Make sure all dependencies are installed
2. **Memory Issues**: Multi-spectral images use more memory; consider reducing resolution if needed
3. **Image Format**: Ensure your 4-band images are properly formatted as 4-channel images

## Future Extensions

Potential improvements:
- Support for more spectral bands
- More sophisticated spectral head architectures
- Spectral-aware densification strategies
- Multi-spectral depth estimation

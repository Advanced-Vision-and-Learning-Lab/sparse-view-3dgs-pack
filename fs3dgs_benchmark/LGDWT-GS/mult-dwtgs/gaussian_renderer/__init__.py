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
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, separate_sh = False, override_color = None, use_trained_exp=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=pipe.antialiasing
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None

    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            if separate_sh:
                dc, shs = pc.get_features_dc, pc.get_features_rest
            else:
                shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    if separate_sh:
        rendered_image, radii, depth_image = rasterizer(
            means3D = means3D,
            means2D = means2D,
            dc = dc,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
    else:
        rendered_image, radii, depth_image = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
        
    # Apply exposure to rendered image (training only)
    if use_trained_exp:
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        rendered_image = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3,   None, None]

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rendered_image = rendered_image.clamp(0, 1)
    
    # Apply spectral head if enabled (disabled for new NIR approach)
    spectral_image = None
    # if pc.use_spectral_head and pc.spectral_head is not None:
    #     # Create a copy to avoid in-place operations
    #     rendered_image_copy = rendered_image.clone()
    #     spectral_image = pc.spectral_head(rendered_image_copy)
    
    # Render NIR if enabled
    nir_image = None
    if pc.use_nir and pc.get_nir_albedo is not None:
        nir_image = render_nir(viewpoint_camera, pc, pipe, bg_color, scaling_modifier, override_color)
    
    out = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter" : (radii > 0).nonzero(),
        "radii": radii,
        "depth" : depth_image
        }
    
    # Add spectral image to output if available
    if spectral_image is not None:
        out["spectral"] = spectral_image
    
    # Add NIR image to output if available
    if nir_image is not None:
        out["nir"] = nir_image
    
    return out

def render_nir(viewpoint_camera, pc, pipe, bg_color, scaling_modifier=1.0, override_color=None):
    """
    Render NIR image using the same rasterization as RGB but with NIR albedo.
    
    Args:
        viewpoint_camera: Camera viewpoint
        pc: Point cloud with NIR albedo
        bg_color: Background color
        scaling_modifier: Scaling modifier
        override_color: Override color (not used for NIR)
    
    Returns:
        torch.Tensor: NIR image of shape (1, H, W)
    """
    # Get NIR albedo
    nir_albedo = pc.get_nir_albedo
    if nir_albedo is None:
        return None
    
    # Normalize NIR albedo shape to (N, 3) for colors_precomp
    # Apply global gain if available
    gain = getattr(pc, "_nir_gain", None)
    nir = nir_albedo if gain is None else nir_albedo * torch.clamp(gain, 0.1, 10.0)
    # Handle common shapes: (N,1), (N,1,1), (N,1,3), (N,3)
    if nir.dim() == 3:
        if nir.shape[1] == 1 and nir.shape[2] == 1:
            nir = nir.squeeze(2).squeeze(1)  # -> (N,)
        elif nir.shape[1] == 1 and nir.shape[2] == 3:
            nir = nir.squeeze(1)  # -> (N,3)
        else:
            nir = nir.view(nir.shape[0], -1)  # fallback
    elif nir.dim() == 2 and nir.shape[1] not in (1, 3):
        nir = nir.view(nir.shape[0], -1)

    if nir.dim() == 1:
        nir = nir.unsqueeze(1)  # -> (N,1)

    if nir.shape[1] == 1:
        nir_colors = nir.repeat(1, 3)  # (N,3)
    elif nir.shape[1] == 3:
        nir_colors = nir  # already (N,3)
    else:
        # Use first channel if unexpected shape, then repeat
        nir_colors = nir[:, :1].repeat(1, 3)
    
    # Use the same rasterization as RGB but with NIR colors
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=pipe.antialiasing
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # Get scaling and rotation / covariance
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
    if override_color is None:
        colors_precomp = None
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii, depth_image = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=None,
        colors_precomp=nir_colors,  # Use NIR colors instead of SH
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp
    )

    # Take only the first channel (since we repeated NIR to 3 channels)
    nir_rendered = rendered_image[0:1, :, :]  # (1, H, W)
    
    return nir_rendered

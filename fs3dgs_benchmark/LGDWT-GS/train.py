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

import os
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from random import randint
from utils.loss_utils import (
    l1_loss, ssim, get_dwt_subbands, charbonnier_loss,
    compute_wef_maps, make_heatmap_rgb, compute_wef_all_subbands, make_wef_grid_image,
    make_wef_grid_image_titled, build_selection_maps
)
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0
    
    # Initialize running-mean ratio scaler for DWT (user-specified scheme)
    dwt_running_mean = 1.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        
        # DWT loss using pytorch-wavelets
        dwt_loss = torch.tensor(0.0, device=image.device)
        if opt.dwt_enable:
            # Ensure batch dimension
            pred_batched = image.unsqueeze(0) if image.dim() == 3 else image
            gt_batched = gt_image.unsqueeze(0) if gt_image.dim() == 3 else gt_image
            
            # Get all DWT subbands
            pred_bands = get_dwt_subbands(pred_batched)
            gt_bands = get_dwt_subbands(gt_batched)
            
            # Compute Charbonnier losses for all subbands
            total_dwt_loss = 0.0
            
            # Level 1 subbands (1/2 resolution)
            if opt.dwt_ll1_weight != 0.0:
                total_dwt_loss += opt.dwt_ll1_weight * charbonnier_loss(pred_bands['LL1'], gt_bands['LL1'])
            if opt.dwt_lh1_weight != 0.0:
                total_dwt_loss += opt.dwt_lh1_weight * charbonnier_loss(pred_bands['LH1'], gt_bands['LH1'])
            if opt.dwt_hl1_weight != 0.0:
                total_dwt_loss += opt.dwt_hl1_weight * charbonnier_loss(pred_bands['HL1'], gt_bands['HL1'])
            if opt.dwt_hh1_weight != 0.0:
                total_dwt_loss += opt.dwt_hh1_weight * charbonnier_loss(pred_bands['HH1'], gt_bands['HH1'])
            
            # Level 2 subbands (1/4 resolution)
            if opt.dwt_ll2_weight != 0.0:
                total_dwt_loss += opt.dwt_ll2_weight * charbonnier_loss(pred_bands['LL2'], gt_bands['LL2'])
            if opt.dwt_lh2_weight != 0.0:
                total_dwt_loss += opt.dwt_lh2_weight * charbonnier_loss(pred_bands['LH2'], gt_bands['LH2'])
            if opt.dwt_hl2_weight != 0.0:
                total_dwt_loss += opt.dwt_hl2_weight * charbonnier_loss(pred_bands['HL2'], gt_bands['HL2'])
            if opt.dwt_hh2_weight != 0.0:
                total_dwt_loss += opt.dwt_hh2_weight * charbonnier_loss(pred_bands['HH2'], gt_bands['HH2'])
            
            dwt_loss = total_dwt_loss
        
        # Compute RGB-based selection maps for reweighting
        sel_maps = build_selection_maps(
            image.unsqueeze(0) if image.dim()==3 else image,
            gt_image.unsqueeze(0) if gt_image.dim()==3 else gt_image,
            a=1.0, b=0.3
        )
        S = sel_maps['S']      # (N,1,H,W)
        # Soft integration weight map W = normalize(S) * (1 - veto)
        # Normalize S per-sample to [0,1]
        S_min = S.amin(dim=(-2, -1), keepdim=True)
        S_max = S.amax(dim=(-2, -1), keepdim=True)
        S_norm = (S - S_min) / (S_max - S_min + 1e-8)
        veto = sel_maps.get('veto', torch.zeros_like(S))
        W_soft = S_norm * (1.0 - veto)
        
        # Build hard mask from top 10% of W_soft (per-sample)
        W_flat = W_soft.view(W_soft.shape[0], -1)
        k = (W_flat.shape[1] * 10) // 100
        k = max(1, int(k))
        topk_vals, _ = torch.topk(W_flat, k, dim=1)
        thr = topk_vals[:, -1:].view(W_soft.shape[0], 1, 1, 1)
        hard_mask = (W_soft >= thr).float()
        
        # No veto for downstream S_hat (per your earlier request) -> use S directly
        S_hat = S
        
        # CVCF: Candidate selection (top fraction of S_hat)
        if opt.cvcf_enable and iteration >= opt.cvcf_start_iter and iteration % opt.cvcf_refresh_interval == 0:
            # Compute candidate mask: top cvcf_candidate_topk fraction of S_hat using top-k approach
            S_hat_flat = S_hat.view(S_hat.shape[0], -1)  # (N, H*W)
            total_pixels = S_hat_flat.shape[1]
            n_candidates = int(opt.cvcf_candidate_topk * total_pixels)
            
            # Get top-k values
            top_k_values, top_k_indices = torch.topk(S_hat_flat, n_candidates, dim=1)
            threshold = top_k_values[:, -1:]  # Last (smallest) of top-k values
            
            # Reshape threshold to match S_hat shape
            threshold = threshold.view(S_hat.shape[0], 1, 1, 1)
            candidates_mask = (S_hat >= threshold).float()  # (N, 1, H, W)
            
            # Get candidate pixel coordinates
            batch_size, channels, height, width = candidates_mask.shape
            candidate_coords = []
            for b in range(batch_size):
                mask_b = candidates_mask[b, 0]  # (H, W)
                y_coords, x_coords = torch.where(mask_b > 0.5)
                coords = torch.stack([x_coords.float(), y_coords.float()], dim=1)  # (N_candidates, 2)
                candidate_coords.append(coords)
            
            # Save CVCF outputs
            cvcf_dir = os.path.join(dataset.model_path, "cvcf")
            os.makedirs(cvcf_dir, exist_ok=True)
            
            # Save candidate mask as PNG
            mask_np = candidates_mask[0, 0].cpu().numpy()  # (H, W)
            mask_img = Image.fromarray((mask_np * 255).astype(np.uint8))
            mask_img.save(os.path.join(cvcf_dir, f"candidates_mask_{iteration:06d}.png"))
            
            # Save candidate coordinates as tensor
            if len(candidate_coords) > 0 and len(candidate_coords[0]) > 0:
                torch.save(candidate_coords[0], os.path.join(cvcf_dir, f"candidates_coords_{iteration:06d}.pt"))
                
                # Simplified CVCF: Direct 2D-guided densification
                try:
                    # Find top-K peaks in S_hat for densification
                    S_hat_flat = S_hat.view(S_hat.shape[0], -1)  # (N, H*W)
                    top_k_values, top_k_indices = torch.topk(S_hat_flat, opt.cvcf_top_k, dim=1)
                    
                    # Convert flat indices to 2D coordinates
                    height, width = S_hat.shape[-2:]
                    y_coords = top_k_indices // width
                    x_coords = top_k_indices % width
                    
                    # Create densification coordinates (batch, 2)
                    densification_coords = torch.stack([x_coords.float(), y_coords.float()], dim=2)  # (N, K, 2)
                    
                    # Save densification coordinates
                    torch.save(densification_coords[0], os.path.join(cvcf_dir, f"densification_coords_{iteration:06d}.pt"))
                    
                    # Log statistics
                    n_densification_points = len(densification_coords[0])
                    
                    if TENSORBOARD_FOUND:
                        tb_writer.add_scalar(f"CVCF/n_densification_points", n_densification_points, iteration)
                        tb_writer.add_scalar(f"CVCF/densification_ratio", n_densification_points / len(candidate_coords[0]), iteration)
                    
                    print(f"[ITER {iteration}] CVCF: Selected {n_densification_points} points for densification (top {opt.cvcf_top_k} from {len(candidate_coords[0])} candidates)")
                    
                    # Actually perform CVCF-guided densification
                    try:
                        # Load the densification coordinates
                        densification_coords = torch.load(os.path.join(cvcf_dir, f"densification_coords_{iteration:06d}.pt"))
                        
                        # Perform CVCF-guided densification with actual depth
                        rendered_depth = render_pkg["depth"] if "depth" in render_pkg else None
                        gaussians.densify_from_cvcf_coordinates(
                            densification_coords, viewpoint_cam, scene.cameras_extent, rendered_depth=rendered_depth, N=opt.cvcf_iso_children
                        )
                        
                        print(f"[ITER {iteration}] CVCF: Densification completed successfully")
                        
                    except Exception as e:
                        print(f"[ITER {iteration}] CVCF: Densification execution failed: {e}")
                    
                except Exception as e:
                    print(f"[ITER {iteration}] CVCF: Densification selection failed: {e}")
            
            # Log to TensorBoard
            if TENSORBOARD_FOUND:
                # Convert mask to tensor with proper shape (1, H, W) for TensorBoard
                mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).float()
                tb_writer.add_image(f"CVCF/candidates_mask", mask_tensor, iteration)
                tb_writer.add_scalar(f"CVCF/n_candidates", len(candidate_coords[0]) if len(candidate_coords) > 0 else 0, iteration)
                tb_writer.add_scalar(f"CVCF/candidate_ratio", opt.cvcf_candidate_topk, iteration)
            
            print(f"[ITER {iteration}] CVCF: Selected {len(candidate_coords[0]) if len(candidate_coords) > 0 else 0} candidates (top {opt.cvcf_candidate_topk*100:.1f}%)")
        
        # Schedule beta (0.4 -> 0.6 from iters 200 to 1000)
        def lerp(a, b, t):
            return a + (b - a) * t
        if iteration < 200:
            beta_now = 0.4
        elif iteration < 1000:
            t = (iteration - 200) / max(1, (1000 - 200))
            beta_now = lerp(0.4, 0.6, t)
        else:
            beta_now = 0.6
        alpha = 0.6
        
        # Upsample S_hat to match image resolution before computing w
        image_h, image_w = image.shape[-2:]
        S_hat_upsampled = F.interpolate(S_hat, size=(image_h, image_w), mode='bilinear', align_corners=False)
        w = alpha + beta_now * S_hat_upsampled
        # Detach, clamp and renormalize per-image to keep mean≈1.0
        w = w.detach()
        w = w.clamp(0.7, 1.3)
        w = w / (w.mean(dim=(2,3), keepdim=True) + 1e-8)

        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        # Base loss: (1-λ)*L1_w + λ*(1-SSIM), where L1_w uses per-pixel weight w
        # w is already at image resolution from the upsampling above
        l1_weighted = (w * torch.abs((image.unsqueeze(0) - gt_image.unsqueeze(0)))).mean()
        base_loss = (1.0 - opt.lambda_dssim) * l1_weighted + opt.lambda_dssim * (1.0 - ssim_value)
        
        # Very soft local L1 over highest-error pixels (top 10% by W_soft)
        abs_err = torch.abs(image.unsqueeze(0) - gt_image.unsqueeze(0))  # (1,3,H,W)
        # Upsample mask to image resolution
        hard_mask_up = F.interpolate(hard_mask, size=abs_err.shape[-2:], mode='nearest')  # (N,1,H,W)
        # Match mask shape to abs_err channels
        hard_mask_c = hard_mask_up.expand(-1, abs_err.shape[1], -1, -1)
        l1_local = (hard_mask_c * abs_err).sum() / (hard_mask_c.sum() + 1e-8)
        base_loss = base_loss + 0.05 * l1_local
        
        # User-specified running-mean ratio scaling for DWT
        if opt.dwt_enable:
            # ratio of magnitudes (detach to avoid graph)
            ratio = (base_loss.detach() / (dwt_loss.detach() + 1e-8)).item() if dwt_loss.requires_grad or isinstance(dwt_loss, torch.Tensor) else 1.0
            dwt_running_mean = 0.95 * dwt_running_mean + 0.05 * ratio
            dwt_scale = float(max(0.1, min(10.0, dwt_running_mean)))
            loss = base_loss + dwt_scale * dwt_loss
        else:
            loss = base_loss

        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            # Log base scalars
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
            
        # Log DWT losses, WEF heatmaps and scaling information
        if tb_writer and opt.dwt_enable:
            tb_writer.add_scalar('train_loss_patches/dwt_total', dwt_loss.item(), iteration)
            # Log current dynamic scale
            tb_writer.add_scalar('train_loss_patches/dwt_scale_factor', dwt_running_mean, iteration)
            if iteration % 100 == 0:
                print(f"[ITER {iteration}] DWT Running-Mean Scaling: scale={dwt_running_mean:.3f}, "
                      f"dwt_raw={dwt_loss.item():.4f}, base_loss={base_loss.item():.4f}, total_loss={loss.item():.4f}")
            
            # WEF logging at 3 uniform iterations across training
            try:
                total_iters = opt.iterations if opt.iterations > 0 else 1000
                wef_checkpoints = {max(1, total_iters//4), max(1, total_iters//2), total_iters}
                if iteration in wef_checkpoints:
                    # Ensure (N,C,H,W)
                    pred_batched = image.unsqueeze(0) if image.dim() == 3 else image
                    gt_batched = gt_image.unsqueeze(0) if gt_image.dim() == 3 else gt_image
                    wef_maps = compute_wef_maps(pred_batched, gt_batched)
                    # Convert to RGB heatmaps
                    hm_ll2 = make_heatmap_rgb(wef_maps['LL2'])
                    hm_lh2 = make_heatmap_rgb(wef_maps['LH2'])
                    hm_hl2 = make_heatmap_rgb(wef_maps['HL2'])
                    hm_wef = make_heatmap_rgb(wef_maps['WEF'])
                    # Log to TensorBoard (take first sample)
                    tb_writer.add_image('WEF/LL2', hm_ll2[0], iteration)
                    tb_writer.add_image('WEF/LH2', hm_lh2[0], iteration)
                    tb_writer.add_image('WEF/HL2', hm_hl2[0], iteration)
                    tb_writer.add_image('WEF/COMBINED', hm_wef[0], iteration)
                    # Also save PNGs under the run folder
                    wef_dir = os.path.join(dataset.model_path, 'wef')
                    os.makedirs(wef_dir, exist_ok=True)
                    def save_png(tchw, name):
                        arr = (tchw.squeeze(0).permute(1,2,0).detach().cpu().clamp(0,1).numpy() * 255).astype(np.uint8)
                        Image.fromarray(arr).save(os.path.join(wef_dir, f"{name}_{iteration}.png"))
                    save_png(hm_ll2, 'wef_ll2')
                    save_png(hm_lh2, 'wef_lh2')
                    save_png(hm_hl2, 'wef_hl2')
                    save_png(hm_wef, 'wef_combined')

                    # Build only the requested subset (LL2, LH2, HL2, COMBINED) as a compact grid
                    all_maps = compute_wef_all_subbands(pred_batched, gt_batched)

                    # Save full 3x3 grid (8 subbands + COMBINED)
                    grid1 = make_wef_grid_image_titled(
                        all_maps,
                        ['LL1','LH1','HL1','HH1','LL2','LH2','HL2','HH2','COMBINED'],
                        tile_cols=3
                    )
                    grid1.save(os.path.join(wef_dir, f"wef_grid_all_{iteration}.png"))

                    # Save compact 2x2 subset (LL2, LH2, HL2, COMBINED)
                    subset_maps = {
                        'LL2': all_maps['LL2'], 'LH2': all_maps['LH2'], 'HL2': all_maps['HL2'], 'COMBINED': all_maps['COMBINED']
                    }
                    grid2 = make_wef_grid_image_titled(subset_maps, ['LL2','LH2','HL2','COMBINED'], tile_cols=2)
                    grid2.save(os.path.join(wef_dir, f"wef_grid_subset_{iteration}.png"))

                    # Selection maps (LL~, M, S, HH1~, veto) saved once per checkpoint
                    sel = build_selection_maps(pred_batched, gt_batched, use_blur_ll2=True, a=1.0, b=0.3)
                    for k in ['LL_t','LH_t','HL_t','M','S','HH1_t','veto']:
                        hm = make_heatmap_rgb(sel[k]) if k != 'veto' else torch.cat([sel[k]]*3, dim=1)
                        save_png(hm, f"sel_{k}")
            except Exception as e:
                print(f"[WARN] WEF logging failed: {e}")

            # Log individual subband losses if they're being used
            if opt.dwt_ll1_weight != 0.0:
                tb_writer.add_scalar('train_loss_patches/dwt_LL1', charbonnier_loss(pred_bands['LL1'], gt_bands['LL1']).item(), iteration)
            if opt.dwt_ll2_weight != 0.0:
                tb_writer.add_scalar('train_loss_patches/dwt_LL2', charbonnier_loss(pred_bands['LL2'], gt_bands['LL2']).item(), iteration)
            if opt.dwt_lh1_weight != 0.0:
                tb_writer.add_scalar('train_loss_patches/dwt_LH1', charbonnier_loss(pred_bands['LH1'], gt_bands['LH1']).item(), iteration)
            if opt.dwt_hl1_weight != 0.0:
                tb_writer.add_scalar('train_loss_patches/dwt_HL1', charbonnier_loss(pred_bands['HL1'], gt_bands['HL1']).item(), iteration)
            if opt.dwt_hh1_weight != 0.0:
                tb_writer.add_scalar('train_loss_patches/dwt_HH1', charbonnier_loss(pred_bands['HH1'], gt_bands['HH1']).item(), iteration)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")

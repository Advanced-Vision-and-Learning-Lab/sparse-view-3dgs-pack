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
import torch
from random import randint
from utils.loss_utils import (
    l1_loss, ssim, get_dwt_subbands, charbonnier_loss,
    make_heatmap_rgb, make_wef_grid_image,
    make_wef_grid_image_titled, compute_elf_map, compute_patch_dwt_loss,
    compute_wef_maps, compute_wef_all_subbands
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
                total_dwt_loss += opt.dwt_ll1_weight * l1_loss(pred_bands['LL1'], gt_bands['LL1'])
            if opt.dwt_lh1_weight != 0.0:
                total_dwt_loss += opt.dwt_lh1_weight * l1_loss(pred_bands['LH1'], gt_bands['LH1'])
            if opt.dwt_hl1_weight != 0.0:
                total_dwt_loss += opt.dwt_hl1_weight * l1_loss(pred_bands['HL1'], gt_bands['HL1'])
            if opt.dwt_hh1_weight != 0.0:
                total_dwt_loss += opt.dwt_hh1_weight * l1_loss(pred_bands['HH1'], gt_bands['HH1'])
            
            # Level 2 subbands (1/4 resolution)
            if opt.dwt_ll2_weight != 0.0:
                total_dwt_loss += opt.dwt_ll2_weight * l1_loss(pred_bands['LL2'], gt_bands['LL2'])
            if opt.dwt_lh2_weight != 0.0:
                total_dwt_loss += opt.dwt_lh2_weight * l1_loss(pred_bands['LH2'], gt_bands['LH2'])
            if opt.dwt_hl2_weight != 0.0:
                total_dwt_loss += opt.dwt_hl2_weight * l1_loss(pred_bands['HL2'], gt_bands['HL2'])
            if opt.dwt_hh2_weight != 0.0:
                total_dwt_loss += opt.dwt_hh2_weight * l1_loss(pred_bands['HH2'], gt_bands['HH2'])
            
            dwt_loss = total_dwt_loss
        
        # Patch-wise DWT Loss (LGDWT-GS)
        patch_loss = torch.tensor(0.0, device=image.device)
        if opt.patch_dwt_enable:
            pred_b = image.unsqueeze(0) if image.dim() == 3 else image
            gt_b = gt_image.unsqueeze(0) if gt_image.dim() == 3 else gt_image
            
            # Use GT for stable ELF map (as per paper Fig 3)
            elf_map = compute_elf_map(gt_b)
            
            # Compute loss on low-ELF patches
            patch_loss = compute_patch_dwt_loss(
                pred_b, gt_b, elf_map, 
                patch_size=opt.patch_size, percentile=opt.patch_percentile,
                lh1_weight=opt.patch_dwt_lh1_weight, hl1_weight=opt.patch_dwt_hl1_weight
            )

        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        # Base loss: (1-λ)*L1 + λ*(1-SSIM)
        base_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
        
        # User-specified running-mean ratio scaling for DWT
        if opt.dwt_enable:
            # ratio of magnitudes (detach to avoid graph)
            ratio = (base_loss.detach() / (dwt_loss.detach() + 1e-8)).item() if dwt_loss.requires_grad or isinstance(dwt_loss, torch.Tensor) else 1.0
            dwt_running_mean = 0.95 * dwt_running_mean + 0.05 * ratio
            dwt_scale = float(max(0.1, min(10.0, dwt_running_mean)))
            loss = base_loss + dwt_scale * dwt_loss
        else:
            loss = base_loss

        # Add Patch-DWT loss
        if opt.patch_dwt_enable:
            loss = loss + opt.patch_dwt_weight * patch_loss

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
        if tb_writer and opt.patch_dwt_enable:
            tb_writer.add_scalar('train_loss_patches/patch_dwt_loss', patch_loss.item(), iteration)
            
        if tb_writer and opt.dwt_enable:
            tb_writer.add_scalar('train_loss_patches/dwt_total', dwt_loss.item(), iteration)
            # Log current dynamic scale
            tb_writer.add_scalar('train_loss_patches/dwt_scale_factor', dwt_running_mean, iteration)
            if iteration % 100 == 0:
                print(f"[ITER {iteration}] DWT Running-Mean Scaling: scale={dwt_running_mean:.3f}, "
                      f"dwt_raw={dwt_loss.item():.4f}, base_loss={base_loss.item():.4f}, total_loss={loss.item():.4f}")
            
            # Log individual subband losses if they're being used
            if opt.dwt_ll1_weight != 0.0:
                tb_writer.add_scalar('train_loss_patches/dwt_LL1', l1_loss(pred_bands['LL1'], gt_bands['LL1']).item(), iteration)
            if opt.dwt_ll2_weight != 0.0:
                tb_writer.add_scalar('train_loss_patches/dwt_LL2', l1_loss(pred_bands['LL2'], gt_bands['LL2']).item(), iteration)
            if opt.dwt_lh1_weight != 0.0:
                tb_writer.add_scalar('train_loss_patches/dwt_LH1', l1_loss(pred_bands['LH1'], gt_bands['LH1']).item(), iteration)
            if opt.dwt_hl1_weight != 0.0:
                tb_writer.add_scalar('train_loss_patches/dwt_HL1', l1_loss(pred_bands['HL1'], gt_bands['HL1']).item(), iteration)
            if opt.dwt_hh1_weight != 0.0:
                tb_writer.add_scalar('train_loss_patches/dwt_HH1', l1_loss(pred_bands['HH1'], gt_bands['HH1']).item(), iteration)

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

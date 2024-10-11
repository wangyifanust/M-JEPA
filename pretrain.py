# --------------------------------------------------------
# Copyright (c) Meta Platforms, Inc.
# All rights reserved.
# --------------------------------------------------------

# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import argparse
import datetime
import json
import yaml
import numpy as np
import os
import time
from pathlib import Path
import random
import copy
from src.masks.utils import apply_masks
from src.utils.tensors import repeat_interleave_batch
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

import timm
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from engine_pretrain import train_one_epoch
from util import load_checkpoint, init_video_model, init_opt
from src.utils.logging import (
    CSVLogger,
    gpu_timer,
    get_logger,
    grad_logger,
    adamw_logger,
    AverageMeter
)

from model_mamp.transformer import Transformer as MAMPTransformer
def import_class(name):
    """Dynamically import a class from a string name."""
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def get_main_args_parser():
    """Main argument parser for training configurations."""
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    
    # Main configuration file
    parser.add_argument('--config', default='./config/ntu60_xsub_joint_pretrain_debug.yaml',
                        help='Path to the main configuration file')
    
    # Training parameters
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * #gpus)')
    parser.add_argument('--epochs', default=400, type=int, help='Total number of training epochs')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='Start epoch')
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations to increase effective batch size under memory constraints')
    
    # Model parameters
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of the model to train')
    parser.add_argument('--model_args', default=dict(), help='Arguments for the model')
    
    # Masking parameters
    parser.add_argument('--mask_ratio', default=0.90, type=float, help='Masking ratio (percentage of removed patches)')
    parser.add_argument('--mask_ratio_inter', default=0.75, type=float, help='Inter masking ratio')
    parser.add_argument('--mask_ratio_intra', default=0.80, type=float, help='Intra masking ratio')
    
    # Motion parameters
    parser.add_argument('--motion_stride', default=1, type=int, help='Stride of motion to be predicted')
    parser.add_argument('--motion_aware_tau', default=0.75, type=float, help='Temperature of motion aware masking')
    
    # Optimizer parameters
    parser.add_argument('--enable_amp', action='store_true', default=False, help='Enable automatic mixed precision')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='Weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='Learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='Base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='Lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='Epochs to warmup learning rate')
    
    # Dataset parameters
    parser.add_argument('--feeder', default='feeder.feeder', help='Data loader to use')
    parser.add_argument('--train_feeder_args', default=dict(), help='Arguments for the training data loader')
    
    # Output and logging
    parser.add_argument('--output_dir', default='./output_dir', help='Path to save outputs')
    parser.add_argument('--log_dir', default='./output_dir', help='Path for TensorBoard logs')
    
    # Device and seed
    parser.add_argument('--device', default='cuda', help='Device to use for training/testing')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    
    # Checkpointing
    parser.add_argument('--resume', default='', help='Path to resume from checkpoint')
    
    # DataLoader parameters
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers')
    parser.add_argument('--pin_mem', action='store_true', help='Pin CPU memory in DataLoader for more efficient transfer to GPU')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    
    # Distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='Number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='URL to set up distributed training')
    
    return parser


def get_additional_args_parser():
    """Additional argument parser for extra configurations."""
    parser = argparse.ArgumentParser()
    
    # Additional configuration file
    parser.add_argument('--fname', type=str, help='Name of the additional config file to load', default='./configs/pretrain/vitl16.yaml')
    
    # Device parameters
    parser.add_argument('--devices', type=str, nargs='+', default=['cuda:0'],
                        help='Devices to use on the local machine')
    
    return parser


def load_yaml_config(config_path):
    """Load YAML configuration file."""
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.load(f, yaml.FullLoader)
    else:
        raise FileNotFoundError(f"Configuration file {config_path} not found.")


def merge_configs(main_config, additional_config):
    """Merge two configuration dictionaries."""
    merged = copy.deepcopy(main_config)
    for key, value in additional_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key].update(value)
        else:
            merged[key] = value
    return merged


def setup_seed(seed, rank=0):
    """Set random seed for reproducibility."""
    seed = seed + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def initialize_model(args, device, cfgs_mask):
    """Initialize the model, predictor, and target encoder."""
    # Dynamically import and instantiate the model
    ModelClass = import_class(args.model)
    model = ModelClass(**args.model_args)
    model.to(device)
    model2 = MAMPTransformer(
        dim_in=3,
        dim_feat=256,
        decoder_dim_feat=256,
        depth=5,
        decoder_depth=5,
        num_heads=8,
        mlp_ratio=4,
        num_frames=120,
        num_joints=25,
        patch_size=1,
        t_patch_size=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        norm_layer=torch.nn.LayerNorm,
        norm_skes_loss=False
    )
    model2.to(device)

    # If using DistributedDataParallel
    # if args.distributed:
    #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)
    #     model_without_ddp = model.module
    # else:
    #     model_without_ddp = model
    if args.distributed:
        model2 = torch.nn.parallel.DistributedDataParallel(model2, device_ids=[args.local_rank], find_unused_parameters=True)
        model_without_ddp = model2.module
    else:
        model_without_ddp = model2
    encoder=model_without_ddp.encoder
    
    # # Set up optimizer with weight decay for specific parameters
    # param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    
    # # Initialize video model components
    # encoder, predictor = init_video_model(
    #     uniform_power=args.model_args.get('uniform_power', True),
    #     use_mask_tokens=args.model_args.get('use_mask_tokens', True),
    #     num_mask_tokens=len(cfgs_mask),
    #     zero_init_mask_tokens=args.model_args.get('zero_init_mask_tokens', True),
    #     device=device,
    #     patch_size=args.model_args.get('patch_size', 16),
    #     num_frames=args.model_args.get('num_frames', 8),
    #     tubelet_size=args.model_args.get('tubelet_size', 2),
    #     model_name=args.model_args.get('model_name', 'vit_large_patch16'),
    #     crop_size=args.model_args.get('crop_size', 224),
    #     pred_depth=args.model_args.get('pred_depth', 1),
    #     pred_embed_dim=args.model_args.get('pred_embed_dim', 256),
    #     use_sdpa=args.model_args.get('use_sdpa', False),
    # )
    
    # Initialize target encoder
    target_encoder = copy.deepcopy(encoder)
    target_encoder.to(device)
    for p in target_encoder.parameters():
        p.requires_grad = False
    
    # Define optimizer
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay
    )
    
    # Define loss scaler for mixed precision
    loss_scaler = NativeScaler()
    
    return model, model_without_ddp, encoder, predictor, target_encoder, optimizer, loss_scaler


def initialize_data_loader(args):
    """Initialize the training data loader."""
    FeederClass = import_class(args.feeder)
    dataset_train = FeederClass(**args.train_feeder_args)
    print(f"Training dataset: {dataset_train}")
    
    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print(f"Sampler_train = {sampler_train}")
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
    
    def worker_init_fn(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        worker_init_fn=worker_init_fn,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    
    return data_loader_train, sampler_train


def initialize_logging(args):
    """Initialize TensorBoard logging."""
    if misc.is_main_process() and args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None
    return log_writer


def compute_loss(z_list, h_list, loss_exp, masks_pred):
    """Compute the JEPA prediction loss."""
    loss = 0.0
    # Compute loss and accumulate for each mask-enc/mask-pred pair
    for zi, hi in zip(z_list, h_list):
        loss += torch.mean(torch.abs(zi - hi) ** loss_exp) / loss_exp
    loss /= len(masks_pred)
    return loss


def regularization_loss(z_list):
    """Compute the regularization loss."""
    return sum([torch.sqrt(zi.var(dim=1) + 0.0001).mean() for zi in z_list]) / len(z_list)


def update_target_encoder(encoder, target_encoder, momentum):
    """Update target encoder parameters using momentum."""
    for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
        param_k.data.mul_(momentum).add_(param_q.data * (1. - momentum))


# def load_clips(udata, masks_enc, masks_pred, device, batch_size, num_clips):
#     """Prepare clips and masks for training."""
#     # Unsupervised video clips
#     # Put each clip on the GPU and concatenate along batch dimension
#     clips = torch.cat([u.to(device, non_blocking=True) for u in udata[0]], dim=0)
    
#     # Put each mask-enc/mask-pred pair on the GPU and reuse the same mask pair for each clip
#     masks_enc_loaded = []
#     masks_pred_loaded = []
#     for me, mp in zip(masks_enc, masks_pred):
#         me = me.to(device, non_blocking=True)
#         mp = mp.to(device, non_blocking=True)
#         me = repeat_interleave_batch(me, batch_size, repeat=num_clips)
#         mp = repeat_interleave_batch(mp, batch_size, repeat=num_clips)
#         masks_enc_loaded.append(me)
#         masks_pred_loaded.append(mp)
#     return clips, masks_enc_loaded, masks_pred_loaded


def train_one_epoch_custom(encoder, predictor, target_encoder, data_loader, optimizer, device, epoch, loss_scaler, args, log_writer=None):
    """Custom training loop for one epoch, incorporating the V-JEPA training logic."""
    encoder.train()
    predictor.train()
    
    logger = get_logger(__name__)
    
    # Initialize meters for tracking metrics
    loss_meter = AverageMeter()
    input_var_meter = AverageMeter()
    input_var_min_meter = AverageMeter()
    jepa_loss_meter = AverageMeter()
    reg_loss_meter = AverageMeter()
    mask_meters = [AverageMeter() for _ in range(len(args.cfgs_mask))]
    gpu_time_meter = AverageMeter()
    wall_time_meter = AverageMeter()
    
    # Extract necessary configurations
    cfgs_loss = args.model_args.get('loss', {})
    loss_exp = cfgs_loss.get('loss_exp', 2.0)
    reg_coeff = cfgs_loss.get('reg_coeff', 0.1)
    mixed_precision = args.enable_amp
    clip_grad = args.model_args.get('clip_grad', None)
    warmup = args.warmup_epochs
    dtype = torch.float16 if mixed_precision else torch.float32
    
    # Assume 'scheduler', 'wd_scheduler', and 'momentum_scheduler' are defined
    # These should be initialized before and passed to this function if necessary
    scheduler = args.scheduler  # Learning rate scheduler
    wd_scheduler = args.wd_scheduler  # Weight decay scheduler
    momentum_scheduler = args.momentum_scheduler  # Momentum scheduler
    scaler = args.scaler  # For mixed precision training
    
    # Start training loop
    for itr, (udata, masks_enc, masks_pred) in enumerate(data_loader):
        itr_start_time = time.time()
        
        try:
            assert len(masks_enc) == len(masks_pred), 'Currently require num encoder masks = num predictor masks'
            
            batch_size = udata[0][0].size(0)  # Assuming udata[0] is a list of tensors
            num_clips = len(udata[0])
            
            # Load clips and masks
            clips, masks_enc_loaded, masks_pred_loaded = load_clips(udata, masks_enc, masks_pred, device, batch_size, num_clips)
            
            # Update mask meters
            for idx, m in enumerate(mask_meters):
                m.update(masks_enc_loaded[idx][0].size(-1))
            
            # Training step
            (loss, loss_jepa, loss_reg, new_lr, new_wd, grad_stats, grad_stats_pred, optim_stats), gpu_etime_ms = gpu_timer(
                train_step, encoder, predictor, target_encoder, clips, masks_enc_loaded, masks_pred_loaded,
                optimizer, scaler, args, epoch, loss_exp, reg_coeff, mixed_precision, clip_grad, warmup,
                scheduler, wd_scheduler, momentum_scheduler, dtype
            )
            
            iter_elapsed_time_ms = (time.time() - itr_start_time) * 1000.
            
            # Update meters
            loss_meter.update(loss)
            jepa_loss_meter.update(loss_jepa)
            reg_loss_meter.update(loss_reg)
            gpu_time_meter.update(gpu_etime_ms)
            wall_time_meter.update(iter_elapsed_time_ms)
            
            # Compute input variance
            input_var = float(clips.view(clips.size(0), -1).var(dim=1).mean().item())
            input_var_min = float(clips.view(clips.size(0), -1).var(dim=1).min().item())
            input_var_meter.update(input_var)
            input_var_min_meter.update(input_var_min)
            
            # Logging
            if log_writer and itr % 10 == 0:
                global_step = epoch * len(data_loader) + itr
                log_writer.add_scalar("loss/train", loss, global_step)
                log_writer.add_scalar("loss/jepa", loss_jepa, global_step)
                log_writer.add_scalar("loss/reg", loss_reg, global_step)
                log_writer.add_scalar("time/gpu_time_ms", gpu_etime_ms, global_step)
                log_writer.add_scalar("time/wall_time_ms", iter_elapsed_time_ms, global_step)
                log_writer.add_scalar("variance/input_var", input_var, global_step)
                log_writer.add_scalar("variance/input_var_min", input_var_min, global_step)
        
        except Exception as e:
            logger.info(f"Error during training iteration {itr}: {e}")
            continue
    
    # Return metrics for potential further processing
    return {
        'train_loss': loss_meter.avg,
        'train_jepa_loss': jepa_loss_meter.avg,
        'train_reg_loss': reg_loss_meter.avg,
        'gpu_time_ms': gpu_time_meter.avg,
        'wall_time_ms': wall_time_meter.avg,
    }


def train_step(encoder, predictor, target_encoder, clips, masks_enc_loaded, masks_pred_loaded,
               optimizer, scaler, args, epoch, loss_exp, reg_coeff, mixed_precision, clip_grad, warmup,
               scheduler, wd_scheduler, momentum_scheduler, dtype):
    """Perform a single training step."""
    # Update learning rate and weight decay
    new_lr = scheduler.step()
    new_wd = wd_scheduler.step()
    
    # Forward pass for target encoder
    def forward_target(c):
        """Compute target representations."""
        with torch.no_grad():
            h = target_encoder(c)
            h = F.layer_norm(h, (h.size(-1),))  # Normalize over feature dimension [B, N, D]
            # Create targets (masked regions of h)
            h_list = apply_masks(h, masks_pred_loaded, concat=False)
            return h_list
    
    # Forward pass for context encoder and predictor
    def forward_context(c, h_list):
        """Compute context representations and predictions."""
        z = encoder(c, masks_enc_loaded)
        z = predictor(z, h_list, masks_enc_loaded, masks_pred_loaded)
        return z
    
    # Compute losses
    loss_jepa, loss_reg = 0.0, 0.0
    with torch.cuda.amp.autocast(dtype=dtype, enabled=mixed_precision):
        h_list = forward_target(clips)
        z_list = forward_context(clips, h_list)
        loss_jepa = compute_loss(z_list, h_list, loss_exp, masks_pred_loaded)  # JEPA prediction loss
        pstd_z = regularization_loss(z_list)  # Predictor variance across patches
        loss_reg = torch.mean(F.relu(1.0 - pstd_z))
    loss = loss_jepa + reg_coeff * loss_reg
    
    # Backward pass and optimization
    enc_norm, pred_norm = 0.0, 0.0
    optimizer.zero_grad()
    if mixed_precision:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
    else:
        loss.backward()
    if (epoch > warmup) and (clip_grad is not None):
        enc_norm = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip_grad)
        pred_norm = torch.nn.utils.clip_grad_norm_(predictor.parameters(), clip_grad)
    if mixed_precision:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    grad_stats = grad_logger(encoder.named_parameters())
    grad_stats.global_norm = float(enc_norm)
    grad_stats_pred = grad_logger(predictor.named_parameters())
    grad_stats_pred.global_norm = float(pred_norm)
    optimizer.zero_grad()
    optim_stats = adamw_logger(optimizer)
    
    # Momentum update of target encoder
    m = next(momentum_scheduler)
    with torch.no_grad():
        update_target_encoder(encoder, target_encoder, m)
    
    return (
        float(loss.item()),
        float(loss_jepa.item()),
        float(loss_reg.item()),
        new_lr,
        new_wd,
        grad_stats,
        grad_stats_pred,
        optim_stats,
    )


def save_checkpoint(epoch, path, encoder, predictor, optimizer, loss_scaler, target_encoder, args):
    """Save the training checkpoint."""
    save_dict = {
        'encoder': encoder.state_dict(),
        'predictor': predictor.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss_scaler': loss_scaler.state_dict(),
        'target_encoder': target_encoder.state_dict(),
        'epoch': epoch,
        'args': vars(args),
    }
    try:
        torch.save(save_dict, path)
        print(f"Checkpoint saved at {path}")
    except Exception as e:
        print(f"Failed to save checkpoint at {path}: {e}")


def main(args, additional_args):
    """Main training function."""
    # Initialize distributed training
    misc.init_distributed_mode(args)
    
    print(f'Job directory: {os.path.dirname(os.path.realpath(__file__))}')
    print(f"Arguments: {args}")
    print(f"Additional Arguments: {additional_args}")
    
    device = torch.device(args.device)
    
    # Set random seed for reproducibility
    setup_seed(args.seed, misc.get_rank())
    
    cudnn.benchmark = True
    
    # Load main configuration file
    main_config = load_yaml_config(args.config)
    
    # Load additional configuration file
    additional_config = load_yaml_config(additional_args.fname)
    
    # Merge configurations
    merged_config = merge_configs(main_config, additional_config)
    
    # Apply merged configuration to args
    for key, value in merged_config.items():
        if isinstance(value, dict):
            setattr(args, key, value)
        else:
            setattr(args, key, value)
    
    # Extract mask configuration
    args.cfgs_mask = merged_config.get('mask', [])
    
    # Initialize data loader
    data_loader_train, sampler_train = initialize_data_loader(args)
    
    # Initialize logging
    log_writer = initialize_logging(args)
    
    # Initialize model, optimizer, and scaler
    model, model_without_ddp, encoder, predictor, target_encoder, optimizer, loss_scaler = initialize_model(args, device, args.cfgs_mask)
    
    # Initialize scalers and schedulers (assuming they are defined in args or elsewhere)
    args.scaler = loss_scaler
    args.scheduler = ...  # Define your learning rate scheduler
    args.wd_scheduler = ...  # Define your weight decay scheduler
    args.momentum_scheduler = ...  # Define your momentum scheduler
    
    # Load checkpoint if resume path is provided
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        model_without_ddp.load_state_dict(checkpoint['encoder'], strict=False)
        predictor.load_state_dict(checkpoint['predictor'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer'])
        if 'loss_scaler' in checkpoint and checkpoint['loss_scaler'] is not None:
            loss_scaler.load_state_dict(checkpoint['loss_scaler'])
        if 'target_encoder' in checkpoint:
            target_encoder.load_state_dict(checkpoint['target_encoder'])
        print(f"Resumed training from checkpoint {args.resume} at epoch {checkpoint['epoch']}")
    
    logger = get_logger(__name__)
    
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    
    for epoch in range(args.start_epoch, args.epochs):
        logger.info('Epoch %d' % (epoch + 1))
        
        if args.distributed:
            sampler_train.set_epoch(epoch)
        
        # Train for one epoch
        train_stats = train_one_epoch_custom(
            encoder, predictor, target_encoder,
            data_loader_train, optimizer, device,
            epoch, loss_scaler, args,
            log_writer=log_writer
        )
        
        # Save checkpoints
        if args.output_dir and (epoch % 20 == 0 or epoch + 1 == args.epochs):
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}.pth.tar")
            save_checkpoint(epoch, checkpoint_path, encoder, predictor, optimizer, loss_scaler, target_encoder, args)
        
        # Log training stats
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        
        if args.output_dir and misc.is_main_process():
            if log_writer:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), "a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'Training time: {total_time_str}')


if __name__ == '__main__':
    # Main argument parser
    main_parser = get_main_args_parser()
    
    # Additional argument parser
    additional_parser = get_additional_args_parser()
    
    # Parse main arguments first
    main_args, remaining = main_parser.parse_known_args()
    
    # Parse additional arguments with the remaining arguments
    additional_args = additional_parser.parse_args(remaining)
    
    # Load main configuration file
    if main_args.config:
        main_config = load_yaml_config(main_args.config)
        main_parser.set_defaults(**main_config)
    
    # Load additional configuration file
    if additional_args.fname:
        additional_config = load_yaml_config(additional_args.fname)
        # additional_args = merge_configs(additional_args, additional_config)
        # Depending on how you want to handle additional_config, it can be merged later
        # In this case, merging is handled in the main function

    # Parse again to include config file defaults
    main_args = main_parser.parse_args()
    
    # Create output directory if it doesn't exist
    if main_args.output_dir:
        Path(main_args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Start the training process
    main(main_args, additional_args)

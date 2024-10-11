# --------------------------------------------------------
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
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
    
    # If using DistributedDataParallel
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)
        model_without_ddp = model.module
    else:
        model_without_ddp = model
    
    # Set up optimizer with weight decay for specific parameters
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    
    # Initialize video model components
    encoder, predictor = init_video_model(
        uniform_power=args.model_args.get('uniform_power', True),
        use_mask_tokens=args.model_args.get('use_mask_tokens', True),
        num_mask_tokens=len(cfgs_mask),
        zero_init_mask_tokens=args.model_args.get('zero_init_mask_tokens', True),
        device=device,
        patch_size=args.model_args.get('patch_size', 16),
        num_frames=args.model_args.get('num_frames', 8),
        tubelet_size=args.model_args.get('tubelet_size', 2),
        model_name=args.model_args.get('model_name', 'vit_large_patch16'),
        crop_size=args.model_args.get('crop_size', 224),
        pred_depth=args.model_args.get('pred_depth', 1),
        pred_embed_dim=args.model_args.get('pred_embed_dim', 256),
        use_sdpa=args.model_args.get('use_sdpa', False),
    )
    
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


def compute_loss(predictor_output, target_output):
    """Compute Mean Squared Error loss."""
    return torch.mean((predictor_output - target_output) ** 2)


def update_target_encoder(encoder, target_encoder, momentum):
    """Update target encoder parameters using momentum."""
    for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
        param_k.data.mul_(momentum).add_(param_q.data * (1. - momentum))


def train_one_epoch_custom(encoder, predictor, target_encoder, data_loader, optimizer, device, epoch, loss_scaler, args, log_writer=None):
    """Custom training loop for one epoch."""
    encoder.train()
    predictor.train()
    
    # Initialize meters for tracking metrics
    loss_meter = AverageMeter()
    jepa_loss_meter = AverageMeter()
    reg_loss_meter = AverageMeter()
    gpu_time_meter = AverageMeter()
    wall_time_meter = AverageMeter()
    
    # Extract loss configuration
    cfgs_loss = args.model_args.get('loss', {})
    loss_exp = cfgs_loss.get('loss_exp', 2.0)
    reg_coeff = cfgs_loss.get('reg_coeff', 0.1)
    
    # Iterate over data loader
    for step, batch in enumerate(data_loader):
        itr_start_time = time.time()
        
        inputs, masks_enc, masks_pred = batch
        
        inputs = inputs.to(device, non_blocking=True)
        masks_enc = [mask.to(device, non_blocking=True) for mask in masks_enc]
        masks_pred = [mask.to(device, non_blocking=True) for mask in masks_pred]
        
        # Forward pass with automatic mixed precision
        with torch.cuda.amp.autocast(enabled=args.enable_amp):
            # Encoder output
            encoder_output = encoder(inputs, masks_enc)
            # Target encoder output
            with torch.no_grad():
                target_output = target_encoder(inputs, masks_pred)
            # Predictor output
            predictor_output = predictor(encoder_output)
            # Compute loss
            loss = compute_loss(predictor_output, target_output)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss_scaler(loss, optimizer, parameters=list(encoder.parameters()) + list(predictor.parameters()))
        
        # Update target encoder
        update_target_encoder(encoder, target_encoder, args.model_args.get('momentum', 0.999))
        
        # Update meters
        loss_meter.update(loss.item())
        jepa_loss_meter.update(loss.item())  # Assuming loss_jepa is the same as loss
        reg_loss_meter.update(0.0)  # Placeholder, update if regularization loss is computed
        gpu_time = (time.time() - itr_start_time) * 1000  # in milliseconds
        wall_time = (time.time() - itr_start_time) * 1000
        gpu_time_meter.update(gpu_time)
        wall_time_meter.update(wall_time)
        
        # Logging
        if log_writer and step % 10 == 0:
            global_step = epoch * len(data_loader) + step
            log_writer.add_scalar("loss/train", loss.item(), global_step)
            log_writer.add_scalar("loss/jepa", loss.item(), global_step)
            log_writer.add_scalar("loss/reg", 0.0, global_step)
            log_writer.add_scalar("time/gpu_time_ms", gpu_time, global_step)
            log_writer.add_scalar("time/wall_time_ms", wall_time, global_step)
    
    # Log epoch metrics
    if log_writer:
        log_writer.add_scalar("epoch_loss/train", loss_meter.avg, epoch)
        log_writer.add_scalar("epoch_loss/jepa", jepa_loss_meter.avg, epoch)
        log_writer.add_scalar("epoch_loss/reg", reg_loss_meter.avg, epoch)
        log_writer.add_scalar("epoch_time/gpu_time_ms", gpu_time_meter.avg, epoch)
        log_writer.add_scalar("epoch_time/wall_time_ms", wall_time_meter.avg, epoch)
    
    # Return metrics for potential further processing
    return {
        'train_loss': loss_meter.avg,
        'train_jepa_loss': jepa_loss_meter.avg,
        'train_reg_loss': reg_loss_meter.avg,
        'gpu_time_ms': gpu_time_meter.avg,
        'wall_time_ms': wall_time_meter.avg,
    }


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
    cfgs_mask = merged_config.get('mask', [])
    
    # Initialize data loader
    data_loader_train, sampler_train = initialize_data_loader(args)
    
    # Initialize logging
    log_writer = initialize_logging(args)
    
    # Initialize model, optimizer, and scaler
    model, model_without_ddp, encoder, predictor, target_encoder, optimizer, loss_scaler = initialize_model(args, device, cfgs_mask)
    
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
    
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    
    for epoch in range(args.start_epoch, args.epochs):
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
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}.pth")
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
        # Depending on how you want to handle additional_config, it can be merged later
        # In this case, merging is handled in the main function
    
    # Parse again to include config file defaults
    main_args = main_parser.parse_args()
    
    # Create output directory if it doesn't exist
    if main_args.output_dir:
        Path(main_args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Start the training process
    main(main_args, additional_args)

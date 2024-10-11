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

import torch
import torch.backends.cudnn as cudnn
import tensorboard
from torch.utils.tensorboard import SummaryWriter
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets

import timm

assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

from engine_pretrain import train_one_epoch
from util import (
    load_checkpoint,
    init_video_model,
    init_opt,
)

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])  # import return model
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--config', default='./config/ntu60_xsub_joint_pretrain_debug.yaml', help='path to the configuration file')

    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--model_args', default=dict(), help='the arguments of model')

    parser.add_argument('--mask_ratio', default=0.90, type=float,
                        help='Masking ratio (percentage of removed patches).')
    
    parser.add_argument('--motion_stride', default=1, type=int,
                        help='stride of motion to be predicted.')
    parser.add_argument('--motion_aware_tau', default=0.75, type=float,
                        help='temperature of motion aware masking.')      
                        
    parser.add_argument('--mask_ratio_inter', default=0.75, type=float,
                        help='Masking ratio inter (percentage of removed patches).')
    parser.add_argument('--mask_ratio_intra', default=0.80, type=float,
                        help='Masking ratio intra (percentage of removed patches).')

    # Optimizer parameters
    parser.add_argument('--enable_amp', action='store_true', default=False,
                        help='Enabling automatic mixed precision')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--feeder', default='feeder.feeder', help='data loader will be used')
    parser.add_argument('--train_feeder_args', default=dict(), help='the arguments of data loader for training')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


parser2 = argparse.ArgumentParser()
parser2.add_argument(
    '--fname', type=str,
    help='name of config file to load',
    default='./configs/pretrain/vitl16.yaml')
parser2.add_argument(
    '--devices', type=str, nargs='+', default=['cuda:0'],
    help='which devices to use on local machine')
argsv2 = parser2.parse_args()
cfgs_mask = argsv2.get('mask')

# -- MODEL
cfgs_model = argsv2.get('model')
model_name = cfgs_model.get('model_name')
pred_depth = cfgs_model.get('pred_depth')
pred_embed_dim = cfgs_model.get('pred_embed_dim')
uniform_power = cfgs_model.get('uniform_power', True)
use_mask_tokens = cfgs_model.get('use_mask_tokens', True)
zero_init_mask_tokens = cfgs_model.get('zero_init_mask_tokens', True)

# -- DATA
# cfgs_data = argsv2.get('data')
# dataset_type = cfgs_data.get('dataset_type', 'videodataset')
# mask_type = cfgs_data.get('mask_type', 'multiblock3d')
# dataset_paths = cfgs_data.get('datasets', [])
# datasets_weights = cfgs_data.get('datasets_weights', None)
# if datasets_weights is not None:
#     assert len(datasets_weights) == len(dataset_paths), 'Must have one sampling weight specified for each dataset'
# batch_size = cfgs_data.get('batch_size')
# num_clips = cfgs_data.get('num_clips')
# num_frames = cfgs_data.get('num_frames')
# tubelet_size = cfgs_data.get('tubelet_size')
# sampling_rate = cfgs_data.get('sampling_rate')
# duration = cfgs_data.get('clip_duration', None)
# crop_size = cfgs_data.get('crop_size', 224)
# patch_size = cfgs_data.get('patch_size')
# pin_mem = cfgs_data.get('pin_mem', False)
# num_workers = cfgs_data.get('num_workers', 1)
# filter_short_videos = cfgs_data.get('filter_short_videos', False)
# decode_one_clip = cfgs_data.get('decode_one_clip', True)
# log_resource_util_data = cfgs_data.get('log_resource_utilization', False)

def main(args):
    print(torch.cuda.device_count())
    # if torch.cuda.is_available():
    #     args.gpu = min(args.gpu, torch.cuda.device_count() - 1)
    # else:
    #     print("No GPUs available, defaulting to CPU.")
    misc.init_distributed_mode(args)
    


    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # Load dataset
    Feeder = import_class(args.feeder)
    dataset_train = Feeder(**args.train_feeder_args)
    print(dataset_train)

    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    def worker_init_fn(worker_id):                                                          
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        worker_init_fn=worker_init_fn,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    
    # define the model
    Model = import_class(args.model)
    model = Model(**args.model_args)

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    
    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    encoder, predictor = init_video_model(
        uniform_power=uniform_power,
        use_mask_tokens=use_mask_tokens,
        num_mask_tokens=len(cfgs_mask),
        zero_init_mask_tokens=zero_init_mask_tokens,
        device=device,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        model_name=model_name,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_embed_dim=pred_embed_dim,
        use_sdpa=use_sdpa,
    )
    target_encoder = copy.deepcopy(encoder)
    # encoder = Encoder(**encoder_args).to(device)
    # predictor = Predictor(**predictor_args).to(device)
    encoder = encoder.to(device)
    predictor = predictor.to(device)
    
    target_encoder = target_encoder.to(device)
    for p in target_encoder.parameters():
        p.requires_grad = False

    # 定义优化器
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay
    )
    loss_scaler = NativeScaler()
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_one_epoch2(
            encoder, predictor, target_encoder, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        if args.output_dir and (epoch % 20 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

def train_one_epoch2(encoder, predictor, target_encoder, data_loader, optimizer, device, epoch, loss_scaler, log_writer=None, args=None):
    encoder.train()
    predictor.train()

    for step, batch in enumerate(data_loader):
        # 获取输入数据和遮罩
        inputs, masks = batch
        inputs = inputs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        # 前向传播
        with torch.cuda.amp.autocast(enabled=args.enable_amp):
            # 编码器输出
            encoder_output = encoder(inputs, masks)
            # 目标编码器输出
            with torch.no_grad():
                target_output = target_encoder(inputs, masks)
            # 预测器输出
            predictor_output = predictor(encoder_output)
            # 计算损失
            loss = compute_loss(predictor_output, target_output)

        # 反向传播和优化器步骤
        optimizer.zero_grad()
        loss_scaler(loss, optimizer, parameters=list(encoder.parameters()) + list(predictor.parameters()))
        # 更新目标编码器（例如使用指数移动平均）
        update_target_encoder(encoder, target_encoder, args.momentum)

# 定义损失函数和目标编码器更新函数
def compute_loss(predictor_output, target_output):
    loss = torch.mean((predictor_output - target_output) ** 2)
    return loss

def update_target_encoder(encoder, target_encoder, momentum):
    for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
        param_k.data.mul_(momentum).add_(param_q.data * (1. - momentum))
if __name__ == '__main__':
    parser = get_args_parser()
    
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_args = yaml.load(f, yaml.FullLoader)
        key = vars(p).keys()
        for k in default_args.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_args)

    args = parser.parse_args()
    
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

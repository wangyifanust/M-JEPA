# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, epochs: int,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter
    ema_start = 0.995  # Starting EMA momentum, e.g., 0.996
    ema_end = 1    # Ending EMA momentum, e.g., 1.0
    num_epochs = epochs
    ipe_scale = 1  # Scaling factor if needed
    ipe = len(data_loader)      # Iterations per epoch
    if num_epochs == 0:
        raise ValueError("num_epochs should be larger than 0")
    if ipe == 0:
        raise ValueError("ipe should be larger than 0")
    
    total_steps = int(ipe * num_epochs * ipe_scale)

    # Initialize the momentum scheduler
    momentum_list = [ema_start + i * (ema_end - ema_start) / total_steps for i in range(total_steps + 1)]
    momentum_scheduler = iter(momentum_list)
    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, _, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.float().to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=args.enable_amp):
            loss, _, _ = model(samples,
                               mask_ratio=args.mask_ratio,
                               motion_stride=args.motion_stride,
                               motion_aware_tau=args.motion_aware_tau)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(11)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=filter(lambda p: p.requires_grad, model.parameters()),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            # EMA 更新教师模型的参数
            # Step 3. Momentum EMA update of target encoder
            with torch.no_grad():
                m = next(momentum_scheduler)
                for param_q, param_k in zip(model.module.student.parameters(), model.module.teacher.parameters()):
                    # Update target encoder parameters
                    # param_k.data.mul_(m).add_(param_q.detach().data, alpha=1 - m)
                    param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)

            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
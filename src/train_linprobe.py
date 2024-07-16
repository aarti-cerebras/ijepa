# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    # -- WARNING: IF DOING DISTRIBUTED TRAINING ON A NON-SLURM CLUSTER, MAKE
    # --          SURE TO UPDATE THIS TO GET LOCAL-RANK ON NODE, OR ENSURE
    # --          THAT YOUR JOBS ARE LAUNCHED WITH ONLY 1 DEVICE VISIBLE
    # --          TO EACH PROCESS
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SLURM_LOCALID']
except Exception:
    pass

import copy
import logging
import sys
import yaml

import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter

import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from src.masks.utils import apply_masks
from src.utils.distributed import (
    init_distributed,
    AllReduce
)
from src.utils.logging import (
    CSVLogger,
    gpu_timer,
    grad_logger,
    AverageMeter)
from src.utils.tensors import repeat_interleave_batch
from src.datasets.imagenet1k import make_imagenet1k

from src.helper_linprobe import (
    load_checkpoint,
    init_model,
    init_opt, 
    init_linprobe, 
    load_checkpoint)
from src.transforms import make_linearprobe_transforms
from torch.utils.data import default_collate
from src.utils.metrics_log import MetricLogger, accuracy

# --
log_timings = True
log_freq = 10
checkpoint_freq = 50
# --

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def main(args, resume_preempt=False):

    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- META
    use_bfloat16 = args['meta']['use_bfloat16']
    model_name = args['meta']['model_name']
    load_model = args['meta']['load_checkpoint'] or resume_preempt
    r_file = args['meta']['read_checkpoint']
    copy_data = args['meta']['copy_data']
    key_load = args["meta"]["key_load"]
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    # -- DATA
    # --
    batch_size = args['data']['batch_size']
    val_batch_size = args["data"]['val_batch_size']
    pin_mem = args['data']['pin_mem']
    num_workers = args['data']['num_workers']
    root_path = args['data']['root_path']
    image_folder = args['data']['image_folder']
    crop_size = args['data']['crop_size']
    patch_size = args['data']['patch_size']
    # --
    
    num_classes = args['model']['num_classes']
    in_features = args['model']['in_features']
    use_batch_norm = args['model']['use_batch_norm']
    batch_norm_kwargs = args['model']["batch_norm_kwargs"]
    layer_norm_kwargs = args['model']['layer_norm_kwargs']
    pooling_type = args['model']['pooling_type']
    layer_idx_list = args['model']['layer_idx_list']

    # -- OPTIMIZATION
    ipe_scale = args['optimization']['ipe_scale']  # scheduler scale factor (def: 1.0)
    wd = float(args['optimization']['weight_decay'])
    num_epochs = args['optimization']['epochs']
    lr = args['optimization']['lr']
    step_size = args['optimization']['step_size']
    gamma = args['optimization']['gamma']

    # -- LOGGING
    folder = args['logging']['folder']
    if rank==0 and not os.path.exists(folder):
        os.makedirs(folder)
    tag = args['logging']['write_tag']

    dump = os.path.join(folder, 'params-ijepa.yaml')
    with open(dump, 'w') as f:
        yaml.dump(args, f)
    # ----------------------------------------------------------------------- #

    try:
        mp.set_start_method('spawn')
    except Exception:
        pass

    # -- init torch distributed backend
    world_size, rank = init_distributed()
    logger.info(f'Initialized (rank/world-size) {rank}/{world_size}')
    if rank > 0:
        logger.setLevel(logging.ERROR)

    # -- log/checkpointing paths
    log_file = os.path.join(folder, f'{tag}_r{rank}.csv')
    save_path = os.path.join(folder, f'{tag}' + '-ep{epoch}.pth.tar')
    latest_path = os.path.join(folder, f'{tag}-latest.pth.tar')
    if rank == 0:
        writer = SummaryWriter(folder)
    load_path = None
    if load_model:
        # load_path = os.path.join(folder, r_file) if r_file is not None else latest_path
        load_path = os.path.join(r_file) if r_file is not None else latest_path

    # -- make csv_logger
    csv_logger = CSVLogger(log_file,
                           ('%d', 'epoch'),
                           ('%d', 'itr'),
                           ('%.5f', 'loss'),
                           ('%d', 'time (ms)'))

    # -- init model
    encoder = init_model(
        device=device,
        patch_size=patch_size,
        crop_size=crop_size,
        model_name=model_name)

    linear_probe = init_linprobe(
        device=device, 
        in_features=in_features, 
        num_classes=num_classes, 
        use_batch_norm=use_batch_norm, 
        pooling_type=pooling_type, 
        batch_norm_kwargs=batch_norm_kwargs, 
        layer_norm_kwargs=layer_norm_kwargs)


    # -- make data transforms

    transform_train, transform_val = make_linearprobe_transforms()

    # -- init data-loaders/samplers
    _, unsupervised_loader, unsupervised_sampler = make_imagenet1k(
            transform=transform_train,
            batch_size=batch_size,
            collator=default_collate,
            pin_mem=pin_mem,
            training=True,
            num_workers=num_workers,
            world_size=world_size,
            rank=rank,
            root_path=root_path,
            image_folder=image_folder,
            copy_data=copy_data,
            drop_last=True)
    ipe = len(unsupervised_loader)


    _, unsupervised_loader_val, unsupervised_sampler_val = make_imagenet1k(
            transform=transform_val,
            batch_size=val_batch_size,
            collator=default_collate,
            pin_mem=pin_mem,
            training=False,
            num_workers=num_workers,
            world_size=world_size,
            rank=rank,
            root_path=root_path,
            image_folder=image_folder,
            copy_data=copy_data,
            drop_last=True)

    # -- init optimizer and scheduler
    optimizer, scaler, scheduler = init_opt(
        linear_probe=linear_probe,
        iterations_per_epoch=ipe,
        start_lr=lr,
        step_size=step_size,
        gamma=gamma,
        wd=wd,
        ipe_scale=ipe_scale,
        use_bfloat16=use_bfloat16)
    encoder = DistributedDataParallel(encoder, static_graph=True)
    linear_probe = DistributedDataParallel(linear_probe, static_graph=True)
    for p in encoder.parameters():
        p.requires_grad = False


    start_epoch = 0
    # -- load training checkpoint
    if load_model:
        encoder, linear_probe, optimizer, scaler, start_epoch = load_checkpoint(
            device=device,
            r_path=load_path,
            model=encoder,
            linear_probe=linear_probe,
            opt=optimizer,
            scaler=scaler,
            key_load=key_load)
        for _ in range(start_epoch*ipe):
            scheduler.step()

    def save_checkpoint(epoch):
        save_dict = {
            'target_encoder': encoder.state_dict(),
            'linear_probe': linear_probe.state_dict(),
            'opt': optimizer.state_dict(),
            'scaler': None if scaler is None else scaler.state_dict(),
            'epoch': epoch,
            'loss': loss_meter.avg,
            'batch_size': batch_size,
            'world_size': world_size,
            'lr': lr
        }
        if rank == 0:
            torch.save(save_dict, latest_path)
            if (epoch + 1) % checkpoint_freq == 0:
                torch.save(save_dict, save_path.format(epoch=f'{epoch + 1}'))

    # -- TRAINING LOOP
    step = 0
    max_accuracy = 0.0
    test_stats = evaluate(unsupervised_loader_val, encoder, linear_probe, layer_idx_list, device)
    print(f"Accuracy of the network on the val dataset test images: {test_stats['acc1']:.1f}%")
    max_accuracy = max(max_accuracy, test_stats["acc1"])
    print(f'Max accuracy: {max_accuracy:.2f}%')

    writer.add_scalar('perf/test_acc1', test_stats['acc1'], global_step=0)
    writer.add_scalar('perf/test_acc5', test_stats['acc5'], global_step=0)
    writer.add_scalar('perf/test_loss', test_stats['loss'], global_step=0)
    for epoch in range(start_epoch, num_epochs):
        logger.info('Epoch %d' % (epoch + 1))

        # -- update distributed-data-loader epoch
        unsupervised_sampler.set_epoch(epoch)

        loss_meter = AverageMeter()

        time_meter = AverageMeter()

        for itr, udata in enumerate(unsupervised_loader):

            def load_imgs():
                # -- unsupervised imgs
                imgs = udata[0].to(device, non_blocking=True)
                labels = udata[1].to(device, non_blocking=True)
                return (imgs, labels)
            imgs, labels = load_imgs()

            def train_step():
                _new_lr = scheduler.step()
                # --


                def forward_context():
                    with torch.no_grad():
                        _, z = encoder(imgs, None, layer_idx_list)
                    return z

                def loss_fn(z, h):
                    loss = F.cross_entropy(z, h)
                    loss = AllReduce.apply(loss)
                    return loss

                # Step 1. Forward
                with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
                    features = forward_context()
                    features = linear_probe(features)
                    loss = loss_fn(features, labels)

                #  Step 2. Backward & step
                if use_bfloat16:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                grad_stats = grad_logger(encoder.named_parameters())
                optimizer.zero_grad()

                return (float(loss), _new_lr, grad_stats)
            (loss, _new_lr, grad_stats), etime = gpu_timer(train_step)
            if rank == 0:
                writer.add_scalar("loss", loss, global_step=step)
                writer.add_scalar("lr", _new_lr, global_step=step)
            if grad_stats is not None and rank == 0:
                writer.add_scalar("grad_start_layer", grad_stats.first_layer, global_step=step)
                writer.add_scalar("grad_end_layer", grad_stats.last_layer, global_step=step)
            loss_meter.update(loss)
            time_meter.update(etime)

            # -- Logging
            def log_stats():
                csv_logger.log(epoch + 1, itr, loss, etime)
                if (itr % log_freq == 0) or np.isnan(loss) or np.isinf(loss):
                    logger.info('[%d, %5d] loss: %.3f '
                                '[wd: %.2e] [lr: %.2e] '
                                '[mem: %.2e] '
                                '(%.1f ms)'
                                % (epoch + 1, itr,
                                   loss_meter.avg,
                                   wd,
                                   _new_lr,
                                   torch.cuda.max_memory_allocated() / 1024.**2,
                                   time_meter.avg))

                    if grad_stats is not None:
                        logger.info('[%d, %5d] grad_stats: [%.2e %.2e] (%.2e, %.2e)'
                                    % (epoch + 1, itr,
                                       grad_stats.first_layer,
                                       grad_stats.last_layer,
                                       grad_stats.min,
                                       grad_stats.max))

            log_stats()
            step += 1
            assert not np.isnan(loss), 'loss is nan'

        # -- Save Checkpoint after every epoch
        logger.info('avg. loss %.3f' % loss_meter.avg)
        save_checkpoint(epoch+1)
        test_stats = evaluate(unsupervised_loader_val, encoder, linear_probe, layer_idx_list, device)
        print(f"Accuracy of the network on the val dataset test images: {test_stats['acc1']:.1f}%")
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f'Max accuracy: {max_accuracy:.2f}%')

        writer.add_scalar('perf/test_acc1', test_stats['acc1'], global_step=epoch+1)
        writer.add_scalar('perf/test_acc5', test_stats['acc5'], global_step=epoch+1)
        writer.add_scalar('perf/test_loss', test_stats['loss'], global_step=epoch+1)

        encoder.train()
        linear_probe.train()

@torch.no_grad()
def evaluate(data_loader, model, linear_probe, layer_idx_list, device):
    criterion = torch.nn.CrossEntropyLoss()
    # switch to evaluation mode
    model.eval()
    linear_probe.eval()

    metric_logger = MetricLogger(delimiter="  ")
    

    for itr, batch in enumerate(data_loader):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            _, output = model(images, None, layer_idx_list)
            output = linear_probe(output)
            loss = criterion(output, target)
        
        print(f"output: {output}")

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]

        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

        # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}, iterations in val dataset: {len_dataset}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss, len_dataset=len(data_loader)))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == "__main__":
    main()

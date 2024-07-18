# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
import sys

import torch
import torch.nn as nn

import src.models.vision_transformer as vit
from src.utils.schedulers import StepLR
from src.utils.tensors import trunc_normal_

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def load_checkpoint(
    device,
    r_path,
    model,
    linear_probe,
    opt,
    scaler,
    key_load="target_encoder",
):
    try:
        logger.info(f'read-path: {r_path}')
        checkpoint = torch.load(r_path, map_location=torch.device('cpu'))
        epoch = checkpoint['epoch']

        pretrained_dict_load = checkpoint[key_load]
        msg = model.load_state_dict(pretrained_dict_load)
        logger.info(f'loaded pretrained {key_load} from epoch {epoch} with msg: {msg}')

        if "linear_probe" in checkpoint:
            dict_load = checkpoint["linear_probe"]
            msg = linear_probe.load_state_dict(dict_load)
            logger.info(f'loaded pretrained `linear_probe` from epoch {epoch} with msg: {msg}')

        # -- loading optimizer
        opt.load_state_dict(checkpoint['opt'])
        if scaler is not None:
            scaler.load_state_dict(checkpoint['scaler'])
        logger.info(f'loaded optimizers from epoch {epoch}')
        del checkpoint

    except Exception as e:
        logger.info(f'Encountered exception when loading checkpoint {e}')
        epoch = 0

    return model, linear_probe, opt, scaler, epoch


def init_model(
    device,
    patch_size=16,
    model_name='vit_base',
    crop_size=224,
):
    encoder = vit.__dict__[model_name](
        img_size=[crop_size],
        patch_size=patch_size)

    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            trunc_normal_(m.weight, std=0.01)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    for m in encoder.modules():
        init_weights(m)

    encoder.to(device)

    logger.info(encoder)
    return encoder

class LinearProbe(nn.Module):
    def __init__(self, in_features, num_classes, use_batch_norm, pooling_type, batch_norm_kwargs=None, layer_norm_kwargs=None):
        super().__init__()
        if use_batch_norm:
            self.head_norm = nn.BatchNorm2d(**batch_norm_kwargs)
        else:
            self.head_norm = nn.Identity()

        self.pool_norm = nn.LayerNorm(**layer_norm_kwargs)
        self.pooling_type = pooling_type

        self.linear_layer = nn.Linear(in_features, num_classes)

    def forward(self, features):
        if self.pooling_type == "avg_pool_patch":
            # average all patch features across all feature_lists
            features = torch.cat(features, dim=1)
            features = torch.mean(features, dim=1)  # shape = (bsz, H)
            features = self.pool_norm(features)
        elif self.pooling_type == "avg_pool_patch_concat":
            # average all patch features per layer and concatenate along hidden dim
            features = [torch.mean(x, dim=1) for x in features]
            features = torch.cat(
                features, dim=1
            )  # shape = (bsz, len(features_list)*H)
            features = self.pool_norm(features)

        features = self.head_norm(features[:,:, None, None])
        features = self.linear_layer(features.squeeze())

        return features


def init_linprobe(
    device, 
    in_features, 
    num_classes, 
    use_batch_norm=True, 
    pooling_type="avg_pool_patch_concat", 
    batch_norm_kwargs=None, 
    layer_norm_kwargs=None
):
    linear_probe = LinearProbe(in_features, 
    num_classes, 
    use_batch_norm, 
    pooling_type, 
    batch_norm_kwargs, 
    layer_norm_kwargs)

    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.BatchNorm2d):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                torch.nn.init.constant_(m.weight, 1.0)

    for m in linear_probe.modules():
        init_weights(m)
    
    linear_probe.to(device)
    logger.info(linear_probe)

    return linear_probe



def init_opt(
    linear_probe,
    iterations_per_epoch,
    start_lr,
    step_size,
    gamma,
    wd=1e-6,
    use_bfloat16=False,
    ipe_scale=1.25
):
    param_groups = [
        {
            'params': (p for n, p in linear_probe.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1)),
            'WD_exclude': False,
            'weight_decay': wd
        }, {
            'params': (p for n, p in linear_probe.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': True,
            'weight_decay': 0
        }
    ]

    logger.info('Using AdamW')
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = StepLR(optimizer, step_size*iterations_per_epoch, start_lr, gamma)

    scaler = torch.cuda.amp.GradScaler() if use_bfloat16 else None
    return optimizer, scaler, scheduler

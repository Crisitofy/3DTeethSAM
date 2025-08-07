# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
import torch
import hydra
from hydra import compose
from hydra.utils import instantiate
from omegaconf import OmegaConf

# 添加 sam2 到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # model 目录
sys.path.insert(0, parent_dir)  # 这样 Python 就能找到 sam2 模块


def build_sam2(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
    **kwargs,
):

    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
        ]
    # Initialize Hydra manually
    hydra.initialize(config_path="configs", version_base=None)

    # Read config and init model
    cfg = compose(config_name=config_file, overrides=hydra_overrides_extra)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def _load_checkpoint(model, ckpt_path):
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)["model"]
        missing_keys, unexpected_keys = model.load_state_dict(sd, strict=False)
        if missing_keys:
            dsp_related = all('dsp' in key for key in missing_keys)
            if dsp_related:
                for name, module in model.named_modules():
                    if 'dsp' in name:
                        if hasattr(module, 'reset_parameters'):
                            module.reset_parameters()
                        elif isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                            torch.nn.init.kaiming_normal_(module.weight)
                            if module.bias is not None:
                                torch.nn.init.zeros_(module.bias)
            else:
                logging.error(missing_keys)
        if unexpected_keys:
            logging.error(unexpected_keys)
        logging.info("dgap load success")

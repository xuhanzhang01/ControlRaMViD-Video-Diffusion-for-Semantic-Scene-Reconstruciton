import sys
import os

import torch
import torch.nn.init as init
from share import *
import argparse
from RaMViD.diffusion_openai.script_util import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    args_to_dict,
    add_dict_to_argparser,
)


def get_node_name(name, parent_name):
    if len(name) <= len(parent_name):
        return False, ''
    p = name[:len(parent_name)]
    if p != parent_name:
        return False, ''
    return True, name[len(parent_name):]

def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=64,
        microbatch=32,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=2000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        clip=1,
        seed=123,
        anneal_type=None,
        steps_drop=0.0,
        drop=0.0,
        decay=0.0,
        seq_len=20,
        max_num_mask_frames=4,
        mask_range=None,
        uncondition_rate=0.0,
        exclude_conditional=True,
        input_path=None,
        output_path=None
    )
    model_defaults = dict(
        image_size=64,
        num_channels=128,
        num_res_blocks=3,
        num_heads=4,
        num_heads_upsample=-1,
        attention_resolutions="16,8",
        dropout=0.0,
        learn_sigma=False,
        sigma_small=False,
        class_cond=False,
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=True,
        rescale_learned_sigmas=True,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        scale_time_dim=0,
        rgb=True,
        hint_channels=3,
    )
    # defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    add_dict_to_argparser(parser,model_defaults)
    return parser


parser = create_argparser()
args = parser.parse_args()

model, diffusion = create_model_and_diffusion(
    **args_to_dict(args, model_and_diffusion_defaults().keys())
)
input_path = args.input_path
output_path = args.output_path
pretrained_weights = torch.load(input_path)
if 'state_dict' in pretrained_weights:
    pretrained_weights = pretrained_weights['state_dict']

scratch_dict = model.state_dict()

target_dict = {}
for k in scratch_dict.keys():
    is_control, name = get_node_name(k, 'control_')
    if is_control:
        copy_k = 'model.diffusion_' + name
    else:
        copy_k = k
    if copy_k in pretrained_weights:
        target_dict[k] = pretrained_weights[copy_k].clone()
    else:
        if k.endswith(".weight") or k.endswith(".bias"):  # Check if it's a weight parameter
            init.normal_(scratch_dict[k], mean=0, std=0.01)  # Initialize with small random values
            target_dict[k] = scratch_dict[k].clone()
            print(f'These weights are newly added: {k}')
        else:
            target_dict[k] = scratch_dict[k].clone()

model.load_state_dict(target_dict, strict=False)
torch.save(model.state_dict(), output_path)
print('Done.')

import argparse
import yaml
from copy import deepcopy
import os
import glob
import math


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_config', type=str,
        help='name of config file to load',
        default='configs.yaml')
    parser.add_argument(
        '--run_dir', type=str,
        help='path to run_dir to store yamls',
        default='configs.yaml')
    parser.add_argument(
        '--num_devices', type=int,
        help='path to run_dir to store yamls',
        default='configs.yaml')
    parser.add_argument(
        '--orig_global_bsz', type=int,
        help='path to run_dir to store yamls',
        default='configs.yaml')
    parser.add_argument(
        '--sweep_per_device_bsz', type=int,
        help='path to run_dir to store yamls',
        default='configs.yaml')
    parser.add_argument(
        '--train_dataset_samples', type=int,
        help='path to run_dir to store yamls',
        default='configs.yaml')

    return parser.parse_args()

cmd = "source /cg3-store/cerebras/venvs/2.3.0/bin/activate \n\n" + \
    "echo \"Running {0} \" \n\n" + \
    "python /cg3-store/cerebras/modelzoos/aarti/r23_ijepa_jul10_wtrainer/cerebras/modelzoo/models/vision/generic_image_encoders/run.py CSX \\\n" + \
    "  --mode train \\\n" + \
    "  --num_csx {1} \\\n" + \
    "  --params {2} \\\n" + \
    "  --python_paths /cg3-store/cerebras/modelzoos/aarti/r23_ijepa_jul10_wtrainer \\\n" + \
    "  --mount_dirs /cg3-store/cerebras \\\n" + \
    "  --model_dir modeldir_{3} \\\n" + \
    "  --mgmt_namespace ganesh-multimodal \\\n" + \
    "  --job_priority p1 \\\n" + \
    "  --job_labels purpose=ijepa_long_runs_{4} \\\n" + \
    "  --disable_version_check \\\n" \



def generate_shell_script(args):

    files_iter = glob.iglob(f"{args.run_dir}/*.yaml")

    for i, file in enumerate(files_iter):
        mdir = os.path.join(args.run_dir, os.path.basename(file).replace(".yaml", ""))
        cmd_yaml = cmd.format(file, args.num_devices, file, mdir, i) 

        print(f"Writing {args.run_dir}/{i}.sh")
        print("-------")
        with open(os.path.join(args.run_dir, f"{i}.sh"), "w") as fh:
            fh.write(cmd_yaml)


def generate_sweep_configs(args, original_yaml):

    global_bsz = args.num_devices * args.sweep_per_device_bsz
    num_steps_per_epoch = float(args.train_dataset_samples/global_bsz)
    print(f"num steps per epoch: {num_steps_per_epoch}")
    
    # Sweep 1
    base_lr_vals = [(0.001, 0.1), (0.001, 0.5), (0.001, 1)]  # (start LR, endLR)
    _lr_scaling = (args.num_devices * args.sweep_per_device_bsz) / args.orig_global_bsz
    lr_vals = [(lr[0] * _lr_scaling, lr[1] * _lr_scaling) for lr in base_lr_vals]
    print(lr_vals)
    
    # Sweep 2
    wd_vals = [0.0005, 0.0]

    # Sweep 3
    num_epochs = [300]
    max_steps = [math.ceil(n_ep * num_steps_per_epoch) for n_ep in num_epochs]

    # Sweep 4
    warmup_epochs = [10]
    warmpup_steps = [math.ceil(n_wep * num_steps_per_epoch) for n_wep in warmup_epochs]

    # Sweep 5
    grad_clip = [True, False]

    num_configs = len(lr_vals) * len(wd_vals) * len(num_epochs) * len(warmup_epochs) * len(grad_clip)
    print(f"NUM_CONFIGS: {num_configs}")

    # print(original_yaml)

    for lr in lr_vals:
        for wd in wd_vals:
            for mx_st in max_steps:
                for wmp_st in warmpup_steps:
                    for gc in grad_clip:
                        run_id = f"ijepa_vith_lr{lr[1]}_wd{wd}_bsz{global_bsz}_steps{mx_st}_wmpst{wmp_st}_gc{gc}"
                        print(f"---- Generating config: ijepa_vith_lr{lr[1]}_wd{wd}_bsz{global_bsz}_steps{mx_st}_wmpst{wmp_st}_gc{gc}")

                        new_yaml = deepcopy(original_yaml)

                        new_yaml["train_input"]["batch_size"] = global_bsz
                        new_yaml["train_input"]["micro_batch_size"] = args.sweep_per_device_bsz

                        new_yaml["model"]["ema"][0]["scheduler_params"]["total_steps"] = mx_st
                        
                        if gc:
                            new_yaml["optimizer"]["max_gradient_norm"] = 1.0
                        else:
                            new_yaml["optimizer"].pop("max_gradient_norm")

                        new_yaml["optimizer"]["weight_decay"] = wd

                        # Linear scheduler
                        new_yaml["optimizer"]["learning_rate"][0]["initial_learning_rate"] = lr[0]
                        new_yaml["optimizer"]["learning_rate"][0]["end_learning_rate"] = lr[1]
                        new_yaml["optimizer"]["learning_rate"][0]["total_iters"] = wmp_st
                        

                        # Cosine decay scheduler
                        new_yaml["optimizer"]["learning_rate"][1]["initial_learning_rate"] = lr[1]
                        new_yaml["optimizer"]["learning_rate"][1]["T_max"] = mx_st - wmp_st


                        ### run config
                        new_yaml["runconfig"]["max_steps"] = mx_st
                        new_yaml["runconfig"]["save_initial_checkpoint"] = True

                        ## wandb
                        new_yaml["wandb"]["run_name"] = run_id
                        new_yaml["wandb"]["run_id"] = run_id

                        new_yaml_path = os.path.join(args.run_dir, f"{run_id}.yaml")
                        with open(new_yaml_path, "w") as fh:
                            yaml.dump(new_yaml, fh, sort_keys=False)
                            print(f"Wrote {new_yaml_path}")


def main(args):

    with open(args.model_config, "r") as fh:
        original_data = yaml.safe_load(fh)
    
    if not os.path.exists(args.run_dir):
        os.makedirs(args.run_dir)

    generate_sweep_configs(args, original_data)

    generate_shell_script(args)


if __name__ == "__main__":
    """
    /cb/home/aarti/envs/env_0618/bin/python sweep_for_csx.py --model_config /cb/home/aarti/mlf/ijepa_fork/configs/params_mono_ijepa.yaml --run_dir /cb/home/aarti/mlf/ijepa_fork/configs_test --num_devices 4 --orig_global_bsz 1960 --sweep_per_device_bsz 490 --train_dataset_samples 1281167
    """
    args = parse_args()
    print(args)


    main(args)
    

    
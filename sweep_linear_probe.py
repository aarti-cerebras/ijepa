import argparse
import yaml
from copy import deepcopy
import os
import glob


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


    return parser.parse_args()

def generate_shell_script(args):

    files_iter = glob.iglob(f"{args.run_dir}/*.yaml")

    cmd = "/home/aarti/mlf/envs/ijepa_0624/bin/python main_linprobe.py --fname {} --devices cuda:0 cuda:1 2>&1 |  tee {}.log"

    for i, file in enumerate(files_iter):

        cmd_yaml = cmd.format(file, file).replace(".yaml.log", ".log")

        with open(os.path.join(args.run_dir, f"{i}.sh"), "w") as fh:
            fh.write(f"echo \"Running {cmd_yaml}\"\n")
            fh.write(cmd_yaml)
        


def generate_sweep_configs(args, original_yaml):
    lr_vals = [0.01, 0.05, 0.001]
    wd_vals = [0.0005, 0.0]
    pooling_types = ["avg_pool_patch_concat", "avg_pool_patch"]
    use_batch_norm_vals = [True, False]
    layer_idx_list = {"avg_pool_patch_concat": [28, 29, 30, 31], "avg_pool_patch": [31]}
    vith_hidden_dim = 1280

    for lr in lr_vals:
        for wd in wd_vals:
            for pool in pooling_types:
                for use_bn in use_batch_norm_vals:
                    run_id = f"ijepa_linprobe_vith_lr{lr}_wd{wd}_pl{pool}_bn{use_bn}"
                    print(f"Generating config: lr:{lr}, wd:{wd}, pool: {pool}, use_bn:{use_bn} -- {run_id}")

                    new_yaml = deepcopy(original_yaml)
                    new_yaml["optimization"]["lr"] = lr
                    new_yaml["optimization"]["weight_decay"] = wd
                    ###
                    new_yaml["model"]["pooling_type"] = pool
                    new_yaml["model"]["layer_idx_list"] = layer_idx_list[pool]
                    new_hidden = len(layer_idx_list[pool]) * vith_hidden_dim
                    new_yaml["model"]["use_batch_norm"] = use_bn


                    if use_bn:
                        new_yaml["model"]["batch_norm_kwargs"]["num_features"] = new_hidden
                    else:
                        new_yaml["model"]["batch_norm_kwargs"] = None

                    new_yaml["model"]["layer_norm_kwargs"]["normalized_shape"] = [new_hidden]
                    new_yaml["model"]["in_features"] = new_hidden 
                    new_yaml["logging"]["folder"] = os.path.join(args.run_dir, run_id)

                    new_yaml_path = os.path.join(args.run_dir, f"{run_id}.yaml")
                    with open(new_yaml_path, "w") as fh:
                        yaml.dump(new_yaml, fh)
                        print(f"Wrote {new_yaml_path}")


def main(args):

    with open(args.model_config, "r") as fh:
        original_data = yaml.safe_load(fh)
    
    if not os.path.exists(args.run_dir):
        os.makedirs(args.run_dir)

    generate_sweep_configs(args, original_data)

    generate_shell_script(args)


if __name__ == "__main__":
    args = parse_args()
    main(args)
    

    
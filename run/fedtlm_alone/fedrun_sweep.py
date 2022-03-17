import os
import sys
import time
import argparse
from loguru import logger
from multiprocessing import Pool

from run_dir import run_dir

from task_config import *
from globalhost import machine_ip

import torch


def run_process(proc):
    os.system(proc)


def add_args(parser):
    parser.add_argument("--tasks", type=str, default="sst_2,agnews")
    parser.add_argument("--model_name", type=str, default="ditilbert")
    parser.add_argument("--machine_name", type=str, default="pclv100")
    parser.add_argument("--client_num", type=int, default=10)
    parser.add_argument("--valid_gpu", type=str, default="0,1")
    parser.add_argument("--vocabulary_type", type=str, default="all")
    parser.add_argument("--niid", type=str, default="-1")
    parser.add_argument("--alpha", type=float, default="1.0")
    return parser


parser = argparse.ArgumentParser()
parser = add_args(parser)
args = parser.parse_args()

# "sst_2"
tasks = ["sst_2", "20news", "agnews"] \
    if args.tasks == "-1" else args.tasks.split(",")
machine_name = args.machine_name
client_num = args.client_num
valid_gpu = args.valid_gpu

run_name = "run/fedtlm_alone/main"
logger.debug(f"run_name: {run_name}")

valid_gpu = ",".join([str(i) for i in range(torch.cuda.device_count())])  \
    if valid_gpu == "-1" else valid_gpu
valid_gpu = valid_gpu.split(",")
n_gpu = len(valid_gpu)
client_device_dict = {i: valid_gpu[i] for i in range(n_gpu)}
logger.warning(f"valid_gpu: {valid_gpu}")

task_config_dir = tlm_task_config_dir
ip = machine_ip[machine_name]

if args.niid == "-1":
    args.niid = "False"
else:
    args.niid = "True"

if machine_name.startswith("pcl"):
    if "124" in machine_name:
        run_dir = "/raid0/zhuo/code/fednlp"
        run_prefix = "/raid0"
    else:
        run_dir = "/raid1/zhuo/code/fednlp"
        run_prefix = "/ghome"
else:
    run_dir = "/data/zhuo/code/fednlp"
    run_prefix = "/data"

model_dict = {
    "distilbert": "distilbert-base-uncased",
    "bert": "bert-base-uncased",
    "albert": "albert-base-v2"
}

# seeds = [0, 42, 911]
# "distilbert", "bert", "albert"
seeds = [42]
all_model = ["distilbert", "albert", "bert"] \
    if args.model_name == "-1" else [args.model_name]

grid_parameters = {
    "20news": {"all_lr": [6e-5, 8e-5, 1e-4],
               "all_epoches": [1, 3]
               },
    "agnews": {"all_lr": [6e-5, 4e-5, 2e-5],
               "all_epoches": [1, 3]
               }
}

do_train = True
if do_train:
    do_test = False
else:
    do_test = True

if do_train:
    all_lr = grid_parameters[tasks[0]]["all_lr"]
    all_epoches = grid_parameters[tasks[0]]["all_epoches"]
else:
    all_lr = [1e-4]
    all_epoches = [1]

logger.info(f"do_train: {do_train}, do_test: {do_test}")

cmds = []
gpu_index = 0
for task in tasks:
    for seed in seeds:
        for epoch in all_epoches:
            for lr in all_lr:
                for model_type in all_model:
                    device_index = gpu_index % n_gpu
                    task_config_dir[task][model_type]["epochs"] = str(epoch)
                    task_config_dir[task][model_type]["lr"] = str(lr)
                    round = task_config_dir[task][model_type]["comm_round"]
                    logger.warning(
                        f"run {task}_model={model_type}_"
                        f"lr={lr}_epoch={epoch}_seed={seed}_round={round} "
                        f"on device: {client_device_dict[device_index]}")
                    cmd = f'CUDA_VISIBLE_DEVICES={client_device_dict[device_index]} python3 {run_dir}/{run_name}.py '
                    options = ["--dataset", task,
                               "--model_type", model_type,
                               "--model_name", model_dict[model_type],
                               "--do_lower_case", "True",
                               "--machine_name", machine_name,
                               "--client_num_in_total", str(client_num),
                               "--wandb_enable", "False",
                               "--niid", args.niid,
                               "--seed", str(seed),
                               "--gpu", "0",
                               "--client_num_in_total", str(args.client_num),
                               "--partition_method", f"niid_label_clients={client_num}_alpha={args.alpha}",
                               "--alpha", str(args.alpha),
                               "--do_train", str(do_train),
                               "--do_test", str(do_test)
                               ]
                    for key, value in task_config_dir[task][model_type].items():
                        options.extend(["--" + key, value])
                    cmd += " ".join(options)
                    cmds.append(cmd)
                    gpu_index += 1

sleep_time = "sleep 3s"
# sleep_time = "sleep 5h"
logger.warning(sleep_time)
run_process(sleep_time)
logger.warning(f"run {len(cmds)} tasks")

pool = Pool(processes=n_gpu)
pool.map(run_process, cmds)

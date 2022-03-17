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
    parser.add_argument("--machine_name", type=str, default="pcl")
    parser.add_argument("--model_name", type=str, default="bilstm")
    parser.add_argument("--client_num", type=int, default=100)
    parser.add_argument("--valid_gpu", type=str, default="0,1")
    parser.add_argument("--vocabulary_type", type=str, default="all")
    parser.add_argument("--niid", type=str, default="-1")
    parser.add_argument("--alpha", type=float, default="1.0")
    return parser


parser = argparse.ArgumentParser()
parser = add_args(parser)
args = parser.parse_args()

# "sst_2", "agnews" "depression", "finance"
tasks = ["sst_2", "agnews", "20news", ] \
    if args.tasks == "-1" else args.tasks.split(",")
machine_name = args.machine_name

valid_gpu = args.valid_gpu

if args.niid == "-1":
    args.niid = "False"
else:
    args.niid = "True"

run_name = "run/fednpm_alone/main"
logger.debug(f"run_name: {run_name}")

valid_gpu = ",".join([str(i) for i in range(torch.cuda.device_count())]) \
    if valid_gpu == "-1" else valid_gpu
valid_gpu = valid_gpu.split(",")
n_gpu = len(valid_gpu)
client_device_dict = {i: valid_gpu[i] for i in range(n_gpu)}
logger.warning(f"valid_gpu: {valid_gpu}")

task_config_dir = stand_task_config_dir
ip = machine_ip[machine_name]

if machine_name.startswith("pcl"):
    if "124" in machine_name:
        run_prefix = "/raid2"
    else:
        run_prefix = "/raid1"
else:
    run_prefix = "/data"

all_models = ["textcnn", "bilstm"] \
    if args.model_name == "-1" else [args.model_name]

grid_parameters = {
    "20news": {"all_lr": [1e-3, 5e-3],
               "all_epoches": [1, 3]
               },
    "agnews": {"all_lr": [1e-3, 5e-3],
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

all_lr = [1e-4]
all_epoches = [1]
seeds = ["0"]
logger.info(f"do_train: {do_train}, do_test: {do_test}")

cmds = []
gpu_index = 0
for task in tasks:
    for model in all_models:
        for seed in seeds:
            for lr in all_lr:
                for epoch in all_epoches:
                    device_index = gpu_index % n_gpu
                    task_config_dir[task]["epochs"] = str(epoch)
                    task_config_dir[task]["lr"] = str(lr)
                    round = task_config_dir[task]["comm_round"]
                    logger.warning(
                        f"run {task}_model={model}_lr={lr}_epoch={epoch}_seed={seed}_round={round}"
                    )
                    cmd = f'CUDA_VISIBLE_DEVICES={client_device_dict[device_index]} python3 {run_prefix}/zhuo/code/fednlp/{run_name}.py '
                    options = ["--dataset", task,
                               "--model", model,
                               "--machine_name", machine_name,
                               "--gpu", "0",
                               "--wandb_enable", "False",
                               "--vocabulary_type", args.vocabulary_type,
                               "--seed", str(seed),
                               "--niid", str(args.niid),
                               "--alpha", str(args.alpha),
                               "--client_num_in_total", str(args.client_num),
                               "--partition_method", f"niid_label_clients={args.client_num}_alpha={args.alpha}",
                               "--do_train", str(do_train),
                               "--do_test", str(do_test)
                               ]
                    for key, value in task_config_dir[task].items():
                        options.extend(["--" + key, value])
                    cmd += " ".join(options)
                    cmds.append(cmd)
                    gpu_index += 1

sleep_time = "sleep 3s"
# sleep_time = "sleep 5h"
logger.warning(sleep_time)
run_process(sleep_time)
logger.warning(f"run {len(cmds)} tasks")

pool = Pool(processes=int(len(cmds)))
pool.map(run_process, cmds)

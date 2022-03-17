import os
import argparse
import random
from copy import deepcopy
import time
import sys
import torch
import wandb
from loguru import logger

from run_dir import run_dir

from globalhost import machine_dict
from training.utils.register import registry
from run.fednpm_alone.config import add_args
from run.fednpm_alone.misc import (init_training_device, setup_seed, report, save_model_parameters,
                                   skip_parameters, get_best_model, test_report, load_model_parameters)
from run.fednpm_alone.model_build_utils import create_model
from run.fednpm_alone.data_utils import load_and_process_dataset
from run.fednpm_alone.fedutils import MySubsetSerialTrainer

from fedlab.utils.serialization import SerializationTool
from fedlab.utils.aggregator import Aggregators


if __name__ == "__main__":
    # parse python script input parameters
    parser = argparse.ArgumentParser()
    args = add_args(parser)

    # logger.info("client start")
    logger.info(f"run script in {run_dir}")
    logger.debug(f"run args: {args}")

    registry.register("logger", logger)

    # set some path
    logger_file_path = os.path.join(
        machine_dict[args.machine_name]["output_logger_path"],
        f"fedtlm_alone_dataset={args.dataset}_seed={args.seed}.log")
    logger.add(open(logger_file_path, "w"))

    args.output_dir = os.path.join(machine_dict[args.machine_name]["output_dir"],
        f"fednpm_alone_{args.dataset}/multi-seed/"
    )
    os.makedirs(args.output_dir, exist_ok=True)

    cached_dir_name = args.model + f"-world_size={args.world_size}"
    args.cache_dir = os.path.join(
        machine_dict[args.machine_name]["cache_dir"],
        cached_dir_name)
    os.makedirs(args.cache_dir, exist_ok=True)

    args.save_dir = os.path.join(args.output_dir, f"model_niid={args.niid}")
    os.makedirs(args.save_dir, exist_ok=True)

    if args.wandb_enable:
        name = "-".join([f"niid={args.niid}",
                         f"lr={args.lr}",
                         f"epoch={args.epochs}",
                         f"optimizer={args.optimizer}",
                         f"nums={args.client_num_in_total}",
                         f"alpha={args.alpha}"])
        project = f"FedAvg-{args.dataset}-{args.model}"
        wandb.init(
            project=project,
            name=name,
            config=args,
            dir=machine_dict[args.machine_name]["wandb_dir"],
            entity="hit-smilelab-fed")
        registry.register("wandb", wandb)

    args.do_skip, line = skip_parameters(args)
    if args.do_skip and args.do_train:
        args.do_train = False
        args.do_train = False
        logger.critical(f"This {line} has finished, please test!")
        exit(0)
    else:
        logger.critical(f"Train with {line} hyper-parameters")

    # Set the random seed.
    setup_seed(args.seed)

    # GPU arrangement: Please customize this function according your own topology.
    args.gpu = init_training_device(args.gpu)
    logger.debug(f"running on: {args.gpu}")

    # load data
    dataset = load_and_process_dataset(args, args.dataset)
    [train_data_num, test_data_num, dev_data_num,
     train_data_global, test_data_global, dev_data_global,
     train_data_local_num_dict, train_data_local_dict,
     dev_data_local_dict, test_data_local_dict,
     source_vocab, target_vocab, embedding_weights] = dataset
    logger.debug(f"train_data_num: {train_data_num}, "
                 f"dev_data_num: {dev_data_num}, "
                 f"test_data_num: {test_data_num}")

    # create model.
    model = create_model(args, model_name=args.model, input_size=len(source_vocab), output_size=len(target_vocab),
        embedding_weights=embedding_weights)
    logger.debug(model)

    clients_num = args.client_num_in_total
    client_id_list = [
        i for i in range(clients_num)
    ]

    ## fedlab setup
    local_model = deepcopy(model)
    if args.ci:
        num_per_round = int(args.client_num_in_total * args.ci)
    else:
        num_per_round = args.client_num_per_in_round
        args.ci = num_per_round / args.client_num_in_total
    logger.critical(f"CI: {args.ci}, Client num per in round: {num_per_round}")

    total_client_num = args.client_num_in_total  # client总数
    aggregator = Aggregators.fedavg_aggregate

    trainer = MySubsetSerialTrainer(
        model=local_model,
        train_dataset=train_data_local_dict,
        test_dataset=None,
        data_slices=client_id_list,
        aggregator=aggregator,
        args=args,
        logger=logger,
    )

    if args.do_train:
        # train procedure
        logger.debug("The procedure is training")
        times = time.strftime("%Y%m%d%H%M%S", time.localtime())
        to_select = [i for i in range(total_client_num)]
        test_global_max_acc = 0.0
        wandb_log = registry.get("wandb")
        for rd in range(args.comm_round):
            model_parameters = SerializationTool.serialize_model(model)
            selection = random.sample(to_select, num_per_round)
            logger.debug(f"selection client({num_per_round}): {selection}")

            aggregated_parameters = trainer.train(
                model_parameters=model_parameters,
                id_list=selection,
                aggregate=True,
                round=rd
            )

            SerializationTool.deserialize_model(model, aggregated_parameters)

            test_acc, test_loss = trainer.test(
                model=model, test_dl=dev_data_local_dict[0]
            )

            logger.critical(f"{args.dataset}-{args.model} "
                            f"train with niid={args.niid}_lr={args.lr}_"
                            f"epoch={args.epochs}_seed={args.seed}_"
                            f"comm_round={args.comm_round}")
            logger.critical(f"Testing "
                            f"Round: {rd}, Current Acc: {round(test_acc, 3)}, "
                            f"Current Loss: {round(test_loss, 3)}, Max Acc: {round(test_global_max_acc, 3)}")
            if wandb_log:
                wandb_log.log({"CurAcc": round(test_acc, 3),
                               "MaxAcc": round(test_global_max_acc, 3)})

            if test_global_max_acc < test_acc:
                test_global_max_acc = test_acc
                save_model_parameters(args, model.state_dict(), times)

        report(args, round(test_global_max_acc, 3), times)

    if args.do_test:
        logger.debug("The procedure is testing")
        times, line = get_best_model(args)
        logger.critical(f"The best parameters is {line}")
        model = load_model_parameters(args, times, model)
        args.eval_batch_size = args.eval_batch_size * 4
        test_acc, test_loss = trainer.test(model=model, test_dl=test_data_global)
        test_report(args, test_acc, line)

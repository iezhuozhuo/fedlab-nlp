import os
import time
import argparse
import random
from copy import deepcopy

import sys
import torch

from fedlab.utils.aggregator import Aggregators
from fedlab.utils.serialization import SerializationTool

run_dir = "/".join(os.path.abspath(sys.argv[0]).split("/")[0:-3])
sys.path.append(run_dir)

import wandb
from loguru import logger
from globalhost import machine_dict
from run.fedtlm_alone.config import add_args
from training.utils.register import registry
from run.fedtlm_alone.misc import (init_training_device, setup_seed, TLMTranier,
                                   report, save_model_parameters, skip_parameters,
                                   get_best_model, load_model_parameters, test_report)
from run.fedtlm_alone.data_utils import load_and_processing_data
from run.fedtlm_alone.model_build_utils import add_model_args, create_model
from run.fedtlm_alone.fedutils import TLMSubsetSerialTrainer


if __name__ == "__main__":
    # parse python script input parameters
    parser = argparse.ArgumentParser()
    args = add_args(parser)

    # logger.info("client start")
    logger.info(f"run script in {run_dir}")
    logger.debug(f"run args: {args}")

    registry.register("logger", logger)
    # registry.register("role", "client")
    # registry.register("rank", args.rank - 1)

    # set some path
    logger_file_path = os.path.join(
        machine_dict[args.machine_name]["output_logger_path"],
        f"fedtlm_alone_dataset={args.dataset}_seed={args.seed}.log")
    logger.add(open(logger_file_path, "w"))

    args.output_dir = os.path.join(machine_dict[args.machine_name]["output_dir"],
        f"fedtlm_alone_{args.dataset}/multi-seed/"
    )
    os.makedirs(args.output_dir, exist_ok=True)

    args.save_dir = os.path.join(args.output_dir, f"model_niid={args.niid}")
    os.makedirs(args.save_dir, exist_ok=True)

    # config wandb
    # lr=5e-05_epoch=1_optimizer=adamw_niid=True_alpha=1.0_num=100_ci=0.1
    if args.wandb_enable:
        name = "-".join([f"niid={args.niid}",
                         f"lr={args.lr}",
                         f"epoch={args.epochs}",
                         f"optimizer={args.optimizer}",
                         f"nums={args.client_num_in_total}",
                         f"alpha={args.alpha}"])
        project = f"FedAvg-{args.dataset}-{args.model_type}"
        wandb.init(
            project=project,
            name=name,
            config=args,
            dir=machine_dict[args.machine_name]["wandb_dir"],
            entity="hit-smilelab-fed")
        registry.register("wandb", wandb)

    args.do_skip, line = skip_parameters(args)
    if args.do_skip:
        args.do_train = False
        args.do_train = False
        logger.critical(f"This {line} has finished, please test!")
        exit(0)
    else:
        logger.debug(f"Train with {line} hyper-parameters")

    # Set the random seed.
    setup_seed(args.seed)

    # GPU arrangement: Please customize this function according your own topology.
    args.gpu = init_training_device(args.gpu)
    logger.debug(f"running on: {args.gpu}")

    # create model.
    model_args = add_model_args(args)
    config, model, tokenizer = create_model(model_args, formulation="classification")
    logger.debug(model)

    clients_num = args.client_num_in_total
    client_id_list = [
        i for i in range(clients_num)
    ]

    # create dataset
    dataset = load_and_processing_data(args, model_args, tokenizer, client_id_list)
    train_data_num, train_data_global, \
    test_data_num, test_data_global, dev_data_num, dev_data_global, \
    train_data_local_num_dict, train_data_local_dict, dev_data_local_num_dict, dev_data_local_dict, \
    test_data_local_num_dict, test_data_local_dict = dataset

    logger.debug(f"train_data_num: {train_data_num}, "
                 f"test_data_num: {test_data_num}, "
                 f"dev_data_num: {dev_data_num}")

    if args.do_train:
        ## fedlab setup
        logger.debug("The procedure is training")
        times = time.strftime("%Y%m%d%H%M", time.localtime())
        local_model = deepcopy(model)
        if args.ci:
            num_per_round = int(args.client_num_in_total * args.ci)
        else:
            num_per_round = args.client_num_per_in_round
            args.ci = num_per_round / args.client_num_in_total
        logger.critical(f"CI: {args.ci}, Client num per in round: {num_per_round}")

        total_client_num = args.client_num_in_total  # client总数
        aggregator = Aggregators.fedavg_aggregate

        Trainer = TLMTranier(
            args=args, device=args.gpu
        )
        trainer = TLMSubsetSerialTrainer(
            model=local_model,
            train_dataset=train_data_local_dict,
            test_dataset=None,
            data_slices=client_id_list,
            aggregator=aggregator,
            args=args,
            logger=logger,
            trainer=Trainer)

        # train procedure
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
                aggregate=True)

            SerializationTool.deserialize_model(model, aggregated_parameters)

            result, _, _ = trainer.trainer.eval_model(
                model=model, test_dl=dev_data_local_dict[0]
            )
            test_acc, test_loss = result["acc"], result["eval_loss"]
            if test_global_max_acc < test_acc:
                test_global_max_acc = test_acc
                save_model_parameters(args, model.state_dict(), times)

            logger.critical(f"{args.dataset}-{args.model_type} "
                            f"train with niid={args.niid}_lr={args.lr}_"
                            f"epoch={args.epochs}_seed={args.seed}_"
                            f"comm_round={args.comm_round}")
            logger.critical(f"Testing "
                            f"Round: {rd}, Current Acc: {round(test_acc, 3)}, "
                            f"Current Loss: {round(test_loss, 3)}, Max Acc: {round(test_global_max_acc, 3)}")
            if wandb_log:
                wandb_log.log({"CurAcc": round(test_acc, 3),
                               "MaxAcc": round(test_global_max_acc, 3)})
        report(args, round(test_global_max_acc, 3), times=times)

    if args.do_test:
        logger.debug("The procedure is testing")
        times, line = get_best_model(args)
        logger.critical(f"The best parameters is {line}")
        model = load_model_parameters(args, times, model)
        args.eval_batch_size = args.eval_batch_size * 4
        trainer = TLMTranier(args=args, device=args.gpu)
        trainer.eval_model(model, test_dl=test_data_global)
        test_report(args, trainer.best_accuracy, line)


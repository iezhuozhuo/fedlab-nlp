import os
import math
import copy
import random
import numpy as np
import sklearn
import json
from sklearn.metrics import matthews_corrcoef, confusion_matrix

import torch
from torch.optim import *
from torch.nn import CrossEntropyLoss
from transformers import get_linear_schedule_with_warmup

from training.utils.register import registry
from data_manager.data_attributes import tc_data_attributes


class TLMTranier():
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.num_labels = tc_data_attributes[args.dataset]
        self.freeze_layers = args.freeze_layers.split(",") if args.freeze_layers else []
        self.results = {}
        self.logger = registry.get("logger")
        self.best_accuracy = 0.0

    def train_model(self, model, train_dl):
        self.logger.info("train_model in device: " + str(self.device))
        model.to(self.device)

        # build optimizer and scheduler
        iteration_in_total = len(
            train_dl) // self.args.gradient_accumulation_steps * self.args.epochs
        optimizer, scheduler = self.build_optimizer(model, iteration_in_total)

        # training result
        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0
        # tr_loss = []
        if self.args.fl_algorithm == "FedProx":
            global_model = copy.deepcopy(model)

        for epoch in range(0, self.args.epochs):
            model.train()
            for batch_idx, batch in enumerate(train_dl):
                batch = tuple(t for t in batch)
                # dataset = TensorDataset(all_guid, all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
                x = batch[1].to(self.device)
                labels = batch[4].to(self.device)

                # (loss), logits, (hidden_states), (attentions)
                output = model(x)
                logits = output[0]

                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

                if self.args.fl_algorithm == "FedProx":
                    fed_prox_reg = 0.0
                    mu = self.args.fedprox_mu
                    for (p, g_p) in zip(model.parameters(), global_model.parameters()):
                        fed_prox_reg += ((mu / 2) * torch.norm((p - g_p.data)) ** 2)
                    loss += fed_prox_reg

                # model outputs are always tuple in pytorch-transformers (see doc)
                # loss = outputs[0]
                current_loss = loss.item()

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                if (batch_idx + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1
            self.logger.info("epoch = %d done, loss = %s" % (epoch, current_loss))

        return global_step, tr_loss / global_step

    def eval_model(self, model, test_dl, device=None):
        if not device:
            device = self.device

        results = {}

        eval_loss = 0.0
        nb_eval_steps = 0
        n_batches = len(test_dl)
        test_sample_len = len(test_dl.dataset)
        preds = np.empty((test_sample_len, self.num_labels))

        out_label_ids = np.empty(test_sample_len)
        model.to(device)
        model.eval()
        self.logger.debug("test_sample_len = %d, n_batches = %d" % (test_sample_len, n_batches))
        for i, batch in enumerate(test_dl):
            with torch.no_grad():
                batch = tuple(t.to(device) for t in batch)

                x = batch[1]
                labels = batch[4]

                output = model(x)
                logits = output[0]

                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                eval_loss += loss.item()

            nb_eval_steps += 1
            start_index = self.args.eval_batch_size * i

            end_index = start_index + self.args.eval_batch_size if i != (n_batches - 1) else test_sample_len
            # logging.info("batch index = %d, start_index = %d, end_index = %d" % (i, start_index, end_index))
            preds[start_index:end_index] = logits.detach().cpu().numpy()
            out_label_ids[start_index:end_index] = labels.detach().cpu().numpy()

        eval_loss = eval_loss / nb_eval_steps

        model_outputs = preds
        preds = np.argmax(preds, axis=1)

        result, wrong = self.compute_metrics(preds, out_label_ids, test_dl.examples)
        result["eval_loss"] = eval_loss
        results.update(result)

        if result["acc"] > self.best_accuracy:
            self.best_accuracy = result["acc"]
        self.logger.debug("best_accuracy = %f" % self.best_accuracy)

        self.results.update(result)
        # self.logger.critical(self.results)
        return result, model_outputs, wrong

    def compute_metrics(self, preds, labels, eval_examples=None):
        assert len(preds) == len(labels)

        extra_metrics = {}
        extra_metrics["acc"] = sklearn.metrics.accuracy_score(labels, preds)
        mismatched = labels != preds

        if eval_examples:
            wrong = [i for (i, v) in zip(eval_examples, mismatched) if v.any()]
        else:
            wrong = ["NA"]

        mcc = matthews_corrcoef(labels, preds)

        tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
        return (
            {**{"mcc": mcc, "tp": tp, "tn": tn, "fp": fp, "fn": fn}, **extra_metrics},
            wrong,
        )

    def build_optimizer(self, model, iteration_in_total):
        warmup_steps = math.ceil(iteration_in_total * self.args.warmup_ratio)
        self.args.warmup_steps = warmup_steps if self.args.warmup_steps == 0 else self.args.warmup_steps
        self.logger.info("warmup steps = %d" % self.args.warmup_steps)

        # freeze exps only apply for distilbert
        if self.args.model_type == "distilbert":
            self.freeze_model_parameters(model)

        if self.args.optimizer == "adamw":
            self.logger.warning("fedtlm_alone Using AdamW as Optimizer")
            optimizer = AdamW(model.parameters(), lr=self.args.lr, eps=self.args.adam_epsilon)
        else:
            self.logger.warning("fedtlm_alone Using SGD as Optimizer")
            optimizer = SGD(model.parameters(), lr=self.args.lr)

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=iteration_in_total
        )
        return optimizer, scheduler

    def freeze_model_parameters(self, model):
        modules = list()
        self.logger.info("freeze layers: %s" % str(self.freeze_layers))
        for layer_idx in self.freeze_layers:
            if layer_idx == "e":
                modules.append(model.distilbert.embeddings)
            else:
                modules.append(model.distilbert.transformer.layer[int(layer_idx)])
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False
        self.logger.info(get_parameter_number(model))


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num/1e6, 'Trainable': trainable_num/1e6}


def setup_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def init_training_device(gpu):
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    return device


def report(args, max_acc, times):
    save_path = os.path.join(args.output_dir, f"model_niid={args.niid}/")
    os.makedirs(save_path, exist_ok=True)
    file = os.path.join(save_path, f"{args.model_type}_sweep_{args.seed}_eval.results")

    with open(file, "a+") as f:
        line = f"time={times}_{args.dataset}_lr={args.lr}_epoch={args.epochs}_" \
               f"optimizer={args.optimizer}_niid={args.niid}_alpha={args.alpha}_" \
               f"num={args.client_num_in_total}_ci={args.ci}_acc={max_acc}"
        f.write(line+"\n")


def save_model_parameters(args, parameters, times):
    save_path = os.path.join(args.output_dir, f"model_niid={args.niid}/")
    os.makedirs(save_path, exist_ok=True)
    file = os.path.join(save_path, f"{args.model_type}_{times}_models.pth")
    torch.save(parameters, file)


def test_report(args, test_acc, line):
    file_json = os.path.join(args.save_dir, f"{args.model_type}_sweep_{args.seed}_test.results")
    with open(file_json, "a+") as file:
        line += f"_test-acc={test_acc}"
        file.write(line+"\n")


def get_best_model(args):
    file = os.path.join(args.save_dir, f"{args.model_type}_sweep_{args.seed}_eval.results")
    max_acc = 0.0
    best_parameters = None
    with open(file) as f:
        for line in f:
            parameters_info = line.strip().split("_")
            for parameter in parameters_info:
                if parameter.startswith("acc"):
                    _, acc = parameter.split("=")
                    if float(acc) > max_acc:
                        max_acc = float(acc)
                        best_parameters = line.strip()
    logger = registry.get("logger")
    logger.warning(f"The best parameters is {best_parameters}")
    times = best_parameters.split("_")[0].split("=")[1]
    return times, best_parameters


def load_model_parameters(args, times, model):
    file = os.path.join(args.save_dir, f"{args.model_type}_{times}_models.pth")
    model_parameters = torch.load(file)
    model.load_state_dict(model_parameters)
    return model


def skip_parameters(args):
    eval_file = os.path.join(args.save_dir, f"{args.model_type}_sweep_{args.seed}_eval.results")
    patten = f"lr={args.lr}_epoch={args.epochs}"

    if not os.path.exists(eval_file):
        return False, patten
    with open(eval_file) as file:
        for line in file:
            if patten in line:
                return True, line
    return False, patten


import os
import torch
import random
import numpy as np

from training.utils.register import registry


def evaluation(test_data_loader, model, gpu, cuda=True, embedder=None):
    logger = registry.get("logger", None)
    model.eval()
    model.to(gpu)
    if embedder:
        embedder.eval()
        embedder.to(gpu)

    test_loss = test_acc = test_total = 0.
    criterion = torch.nn.CrossEntropyLoss().to(gpu)
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_data_loader):
            x = torch.tensor(batch_data["X"])
            y = torch.tensor(batch_data["Y"])
            seq_lens = torch.tensor(batch_data["seq_lens"])
            if cuda:
                x = x.to(device=gpu)
                y = y.to(device=gpu)
                seq_lens = seq_lens.to(device=gpu)
            if embedder:
                x = embedder(x)
            prediction = model(x, batch_size=x.size()[0],
                               seq_lens=seq_lens, device=gpu)
            loss = criterion(prediction, y)
            num_corrects = torch.sum(torch.argmax(prediction, 1).eq(y))

            test_acc += num_corrects.item()
            test_loss += loss.item() * y.size(0)
            test_total += y.size(0)

    test_acc = test_acc / test_total
    test_loss = test_loss / test_total
    if logger:
        logger.debug(f"evaluation num is {test_total}")
    return test_acc, test_loss


def init_training_device(gpu):
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    return device


def report(args, max_acc, times=None):
    file = os.path.join(args.save_dir, f"{args.model}_sweep_{args.seed}_eval.results")
    with open(file, "a+") as f:
        line = f"times={times}_{args.dataset}_lr={args.lr}_epoch={args.epochs}_round={args.comm_round}_" \
               f"niid={args.niid}_alpha={args.alpha}_num={args.client_num_in_total}_ci={args.ci}_acc={max_acc}"
        f.write(line+"\n")


def setup_seed(seed: int):
    # The np.random seed determines the dataset partition.
    # The torch_manual_seed determines the initial weight.
    # We fix these two, so that we can reproduce the result.
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True


def save_model_parameters(args, parameters, times):
    file = os.path.join(args.save_dir, f"{args.model}_{times}_models.pth")
    torch.save(parameters, file)


def skip_parameters(args):
    eval_file = os.path.join(args.save_dir, f"{args.model}_sweep_{args.seed}_eval.results")
    patten = f"lr={args.lr}_epoch={args.epochs}"

    if not os.path.exists(eval_file):
        return False, patten
    with open(eval_file) as file:
        for line in file:
            if patten in line:
                return True, line
    return False, patten


def load_model_parameters(args, times, model):
    file = os.path.join(args.save_dir, f"{args.model_type}_{times}_models.pth")
    model_parameters = torch.load(file)
    model.load_state_dict(model_parameters)
    return model


def get_best_model(args):
    file = os.path.join(args.save_dir, f"{args.model}_sweep_{args.seed}_eval.results")
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


def test_report(args, test_acc, line):
    file_json = os.path.join(args.save_dir, f"{args.model_type}_sweep_{args.seed}_test.results")
    with open(file_json, "a+") as file:
        line += f"_test-acc={test_acc}"
        file.write(line+"\n")
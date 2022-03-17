import os
import random
import threading

import torch
from torch.optim import *

from fedlab.utils.aggregator import Aggregators
from fedlab.core.server.handler import ParameterServerBackendHandler
from fedlab.utils.message_code import MessageCode
from fedlab.core.client.trainer import ClientTrainer
from fedlab.utils.serialization import SerializationTool
from fedlab.core.communicator.processor import Package, PackageProcessor
from fedlab.core.client.manager import ClientManager, ClientPassiveManager
from fedlab.core.client import SERIAL_TRAINER, ORDINARY_TRAINER
from fedlab.core.server import ServerSynchronousManager

from run.fednpm_alone.misc import evaluation
from training.utils.register import registry


class MySyncParameterServerHandler(ParameterServerBackendHandler):

    def __init__(self,
                 model,
                 global_test_data,
                 global_round=5,
                 cuda=True,
                 sample_ratio=1.0,
                 logger=None,
                 gpu=None):
        super(MySyncParameterServerHandler, self).__init__(model, cuda)

        self.global_test_data = global_test_data
        self.gpu = gpu

        self._LOGGER = logger
        self.wandb = registry.get("wandb")

        if sample_ratio < 0.0 or sample_ratio > 1.0:
            raise ValueError("Invalid select ratio: {}".format(sample_ratio))

        # basic setting
        self.client_num_in_total = 0
        self.sample_ratio = sample_ratio

        # client buffer
        self.client_buffer_cache = []
        self.cache_cnt = 0

        # stop condition
        self.global_round = global_round
        self.round = 0

        # test global metrics
        self.test_global_max_acc = 0.

    def stop_condition(self) -> bool:
        return self.round >= self.global_round

    def sample_clients(self):
        selection = random.sample(range(self.client_num_in_total),
                                  self.client_num_per_round)
        return selection

    def add_model(self, sender_rank, model_parameters):

        self.client_buffer_cache.append(model_parameters.clone())
        self.cache_cnt += 1

        # cache is full
        if self.cache_cnt == self.client_num_per_round:
            self._update_model(self.client_buffer_cache)
            self.round += 1
            # TODO
            self.test_on_server()
            return True
        else:
            return False

    def _update_model(self, model_parameters_list):
        self._LOGGER.debug(
            "Model parameters aggregation, number of aggregation elements {}".
                format(len(model_parameters_list)))
        # use aggregator
        serialized_parameters = Aggregators.fedavg_aggregate(
            model_parameters_list)
        SerializationTool.deserialize_model(self._model, serialized_parameters)

        # reset cache cnt
        self.cache_cnt = 0
        self.client_buffer_cache = []
        self.train_flag = False

    @property
    def client_num_per_round(self):
        return max(1, int(self.sample_ratio * self.client_num_in_total))

    def test_on_server(self):
        test_acc, test_loss = evaluation(self.global_test_data, self._model,
                                         self.gpu, self.cuda)
        if self.test_global_max_acc < test_acc:
            self.test_global_max_acc = test_acc
        self._LOGGER.critical(f"Server Testing "
                              f"Round: {self.round}, Current Acc: {round(test_acc, 3)}, "
                              f"Current Loss: {round(test_loss, 3)}, Max Acc: {round(self.test_global_max_acc, 3)}")
        if self.wandb:
            self.wandb.log({"server_GlobalTestLoss": round(test_loss, 3),
                            "server_GlobalTestAcc": round(test_acc, 3),
                            "server_GlobalMaxAcc": round(self.test_global_max_acc, 3)})


class ClientsTrainer(ClientTrainer):

    def __init__(self,
                 model,
                 train_data_loader,
                 test_data_loader,
                 epochs,
                 optimizer,
                 gpu,
                 cuda=True,
                 logger=None):
        super(ClientsTrainer, self).__init__(model, cuda)

        self._train_data_loader = train_data_loader
        self._test_data_loader = test_data_loader
        self.epochs = epochs
        self.optimizer = optimizer
        self.gpu = gpu
        self.rank = registry.get("rank") if registry.get("rank", None) else "-1"
        self.max_acc = 0.
        self._LOGGER = logger

    def train(self, model_parameters=None) -> None:
        SerializationTool.deserialize_model(
            self._model, model_parameters)  # load parameters
        self._model.to(self.gpu)

        criterion = torch.nn.CrossEntropyLoss().to(self.gpu)

        self._LOGGER.info("Local train procedure is running")

        epoch_loss = []
        epoch_acc = []
        for epoch in range(self.epochs):
            self._model.train()
            batch_loss = []
            batch_acc = []
            for batch_idx, batch_data in enumerate(self._train_data_loader):
                x = torch.tensor(batch_data["X"])
                y = torch.tensor(batch_data["Y"])
                seq_lens = torch.tensor(batch_data["seq_lens"])
                if self.cuda:
                    x = x.to(device=self.gpu)
                    y = y.to(device=self.gpu)
                    seq_lens = seq_lens.to(device=self.gpu)
                self.optimizer.zero_grad()
                prediction = self._model(x, x.size()[0], seq_lens, self.gpu)
                loss = criterion(prediction, y)
                num_corrects = torch.sum(torch.argmax(prediction, 1).eq(y))
                acc = 100.0 * num_corrects / x.size()[0]
                loss.backward()
                self.optimizer.step()

                batch_loss.append(loss.item())
                batch_acc.append(acc.item())
                if len(batch_loss) > 0:
                    epoch_loss.append(sum(batch_loss) / len(batch_loss))
                    epoch_acc.append(sum(batch_acc) / len(batch_acc))

            if len(epoch_loss) > 0:
                self._LOGGER.info('Client id: {} Local Training Epoch: {} '
                                  'Loss: {:.3f} Accuracy: {:.3f}'.format(self.rank, epoch,
                                                                         sum(epoch_loss) / len(epoch_loss),
                                                                         sum(epoch_acc) / len(epoch_acc)))
            else:
                self._LOGGER.critical('Client id {}'
                                      'has {} epoch_loss'.format(self.rank, epoch, len(epoch_loss)))

        self._LOGGER.info("Local train procedure is finished")
        return self.model_parameters

    def test(self):

        test_acc, test_loss = evaluation(self._test_data_loader, self._model,
                                         self.gpu, self.cuda)

        if self.max_acc < test_acc:
            self.max_acc = test_acc
            registry.register("local_best_eval", round(self.max_acc, 1))
        return test_acc, test_loss


class MyClientManager(ClientManager):
    def __init__(self, network, trainer, logger=None):
        super().__init__(network, trainer)
        self._LOGGER = logger

    def main_loop(self):
        while True:
            sender_rank, message_code, payload = PackageProcessor.recv_package(src=0)
            if message_code == MessageCode.Exit:
                break
            elif message_code == MessageCode.ParameterUpdate:
                model_parameters = payload[0]
                self._trainer.train(model_parameters=model_parameters)
                test_acc, test_loss = self._trainer.test()
                self._LOGGER.debug('Client {} Test Metrics: '
                                   'CurrentLoss: {:.3f}, CurrentAccuracy: {:.3f}, MaxLocalAccuracy: {:.3f}'.format(
                    self._trainer.rank, test_loss, test_acc, self._trainer.max_acc))
                self.synchronize()
            else:
                raise ValueError("Invalid MessageCode {}. Please see MessageCode Enum".format(message_code))

    def synchronize(self):
        """Synchronize local model with server"""
        self._LOGGER.info("synchronize model parameters with server")
        model_parameters = self._trainer.model_parameters
        pack = Package(message_code=MessageCode.ParameterUpdate,
                       content=model_parameters)
        PackageProcessor.send_package(pack, dst=0)


# my MySerializationTool
class MySerializationTool(object):
    def serialize(self,obj):
        if isinstance(obj, list):
            m_parameters = None
        else:
            m_parameters = self.serialize_model(obj)
        return m_parameters

    def deserialize(self, model, serialized_parameters, mode):
        if isinstance(serialized_parameters, tuple):
            self.deserialize_parameters(model, serialized_parameters, mode)
        else:
            self.deserialize_model(model, serialized_parameters, mode)

    @staticmethod
    def serialize_model(model: torch.nn.Module) -> torch.Tensor:
        parameters = [param.data.view(-1) for param in model.parameters()]
        m_parameters = torch.cat(parameters)
        m_parameters = m_parameters.cpu()
        return m_parameters

    @staticmethod
    def deserialize_model(model: torch.nn.Module,
                          serialized_parameters: torch.Tensor,
                          mode="copy"):

        current_index = 0  # keep track of where to read from grad_update
        for parameter in model.parameters():
            numel = parameter.data.numel()
            size = parameter.data.size()
            if mode == "copy":
                parameter.data.copy_(
                    serialized_parameters[current_index:current_index +
                                                        numel].view(size))
            elif mode == "add":
                parameter.data.add_(
                    serialized_parameters[current_index:current_index +
                                                        numel].view(size))
            else:
                raise ValueError(
                    "Invalid deserialize mode {}, require \"copy\" or \"add\" "
                        .format(mode))
            current_index += numel

    @staticmethod
    def serialize_parameters(parameters: list) -> tuple:
        params = [param.data.view(-1) for (_, param) in parameters]
        m_parameters = torch.cat(params)
        m_parameters = m_parameters.cpu()
        m_name = [name for (name, _) in parameters]
        return (m_name, m_parameters)

    @staticmethod
    def deserialize_parameters(model: torch.nn.Module,
                               serialized_parameters: tuple,
                               mode="copy"):
        current_index = 0  # keep track of where to read from grad_update
        m_name, m_parameters = serialized_parameters
        for name, parameter in model.named_parameters():
            if name in m_name:
                numel = parameter.data.numel()
                size = parameter.data.size()
                if mode == "copy":
                    parameter.data.copy_(
                        m_parameters[current_index:current_index +
                                                   numel].view(size))
                elif mode == "add":
                    parameter.data.add_(
                        m_parameters[current_index:current_index +
                                                   numel].view(size))
                else:
                    raise ValueError(
                        "Invalid deserialize mode {}, require \"copy\" or \"add\" ".format(mode))
                current_index += numel
            else:
                print(f"{name} not agg")


#### Scale Training
class MySerialTrainer(ClientTrainer):
    def __init__(self,
                 model,
                 client_num,
                 aggregator=None,
                 cuda=True,
                 logger=None):
        super().__init__(model, cuda)
        self.client_num = client_num
        self.type = SERIAL_TRAINER  # represent serial trainer
        self.aggregator = aggregator
        self._LOGGER = logger

    def _train_alone(self, model_parameters, train_loader, indx, round=None):
        """Train local model with :attr:`model_parameters` on :attr:`train_loader`.

        Args:
            model_parameters (torch.Tensor): Serialized model parameters of one model.
            train_loader (torch.utils.data.DataLoader): :class:`torch.utils.data.DataLoader` for this client.
        """
        raise NotImplementedError()

    def _get_dataloader(self, dataset, client_id: int):
        """Get :class:`DataLoader` for ``client_id``."""
        raise NotImplementedError()

    def train(self, model_parameters, id_list, aggregate=False, round=None):
        param_list = []
        self._LOGGER.info(
            "Local training with client id list: {}".format(id_list))
        for idx in id_list:
            self._LOGGER.info(
                "Starting training procedure of client [{}]".format(idx))

            train_data_loader = self._get_dataloader(dataset=self.train_dataset, client_id=idx)
            self._train_alone(model_parameters=model_parameters,
                              train_loader=train_data_loader,
                              indx=idx, round=round)
            # test_data_loader = self._get_dataloader(dataset=self.test_dataset, client_id=idx)
            # test_acc, test_loss, local_max_acc = self._test_alone(test_data_loader, idx)
            # self._LOGGER.debug('Client {} Test Metrics: '
            #                    'CurrentLoss: {:.3f}, CurrentAccuracy: {:.3f}, MaxLocalAccuracy: {:.3f}'.format(
            #     idx, test_loss, test_acc, local_max_acc))

            param_list.append(self.model_parameters)

        if aggregate is True and self.aggregator is not None:
            # aggregate model parameters of this client group
            aggregated_parameters = self.aggregator(param_list)
            return aggregated_parameters
        else:
            return param_list

    def test(self, model, test_dl):
        test_acc, test_loss = evaluation(test_dl, model,
            self.args.gpu, self.cuda)
        return test_acc, test_loss


class MySubsetSerialTrainer(MySerialTrainer):
    def __init__(self,
                 model,
                 train_dataset,
                 test_dataset,
                 data_slices,
                 aggregator=None,
                 logger=None,
                 cuda=True,
                 args=None,
                 embedder=None) -> None:

        super(MySubsetSerialTrainer, self).__init__(model=model,
                                                    client_num=len(data_slices),
                                                    cuda=cuda,
                                                    aggregator=aggregator,
                                                    logger=logger)

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.data_slices = data_slices  # [0, client_num)
        self.args = args
        self.embedder = embedder

        self.wandb = registry.get("wandb")

    def _get_dataloader(self, dataset, client_id: int):
        if isinstance(dataset, dict):
            train_loader = dataset[client_id]
        else:
            train_loader = dataset
        return train_loader

    def _train_alone(self, model_parameters, train_loader, indx, round=None):
        SerializationTool.deserialize_model(self._model, model_parameters)
        criterion = torch.nn.CrossEntropyLoss().to(self.args.gpu)
        # filter(lambda x: x.requires_grad, self._model.parameters()),
        # optimizer = Adam([{"params": filter(lambda x: x.requires_grad, self._model.parameters())}],
        #                  lr=self.args.lr, weight_decay=self.args.wd)

        if round and self.args.embedding_lr:
            # self._LOGGER.critical(f"word embedding layer with lr {self.args.embedding_lr}")
            updates_round = 0.1 * self.args.comm_round
            for parameter in self._model.word_embeddings.parameters():
                if round < updates_round:
                    self._LOGGER.warning(f"word embedding layer with lr {self.args.embedding_lr}")
                    parameter.requires_grad = True
                else:
                    parameter.requires_grad = False
        self.args.embedding_lr = self.args.embedding_lr if self.args.embedding_lr else self.args.lr

        if self.args.model == "bilstm":
            update_params = [
                {"params": filter(lambda x: x.requires_grad, self._model.lstm_layer.parameters()),
                 "lr": self.args.lr, "weight_decay": self.args.wd},
                {"params": filter(lambda x: x.requires_grad, self._model.output_layer.parameters()),
                 "lr": self.args.lr, "weight_decay": self.args.wd},
                {"params": filter(lambda x: x.requires_grad, self._model.word_embeddings.parameters()),
                 "lr": self.args.embedding_lr, "weight_decay": self.args.embedding_wd}
            ]
        else:
            update_params = [
                {"params": filter(lambda x: x.requires_grad, self._model.convs.parameters()),
                 "lr": self.args.lr, "weight_decay": self.args.wd},
                {"params": filter(lambda x: x.requires_grad, self._model.fc.parameters()),
                 "lr": self.args.lr, "weight_decay": self.args.wd},
                {"params": filter(lambda x: x.requires_grad, self._model.word_embeddings.parameters()),
                 "lr": self.args.embedding_lr, "weight_decay": self.args.embedding_wd}
            ]

        optimizer = Adam(update_params)

        if self.cuda:
            self._model.to(device=self.args.gpu)

        epoch_loss, epoch_acc = [], []
        for epoch in range(self.args.epochs):
            self._model.train()

            batch_loss = []
            batch_acc = []
            for batch_idx, batch_data in enumerate(train_loader):
                x = torch.tensor(batch_data["X"])
                y = torch.tensor(batch_data["Y"])
                seq_lens = torch.tensor(batch_data["seq_lens"])
                if self.cuda:
                    x = x.to(device=self.args.gpu)
                    y = y.to(device=self.args.gpu)
                    seq_lens = seq_lens.to(device=self.args.gpu)
                optimizer.zero_grad()
                prediction = self._model(x, batch_size=x.size()[0],
                                         seq_lens=seq_lens, device=self.args.gpu)
                loss = criterion(prediction, y)
                num_corrects = torch.sum(torch.argmax(prediction, 1).eq(y))
                acc = num_corrects / x.size()[0]
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())
                batch_acc.append(acc.item())
                if len(batch_loss) > 0:
                    epoch_loss.append(sum(batch_loss) / len(batch_loss))
                    epoch_acc.append(sum(batch_acc) / len(batch_acc))

            if len(epoch_loss) > 0:
                self._LOGGER.info('Client id: {}, Local Training Epoch: {}, '
                                  'Loss: {:.3f}, Accuracy: {:.3f}'.format(indx, epoch,
                                                                         sum(epoch_loss) / len(epoch_loss),
                                                                         sum(epoch_acc) / len(epoch_acc)))
                # self.wandb.log({f"client_{indx}_LocalTrainLoss": sum(epoch_loss) / len(epoch_loss),
                #                 f"client_{indx}_LocalTrainAcc": sum(epoch_acc) / len(epoch_acc)})
            else:
                self._LOGGER.critical('Client id {}'
                                      'has {} epoch_loss'.format(indx, epoch, len(epoch_loss)))

        self._LOGGER.info(f"Client id {indx} train procedure is finished")
        return self.model_parameters

    def _test_alone(self, test_loader, indx):
        # gpu = init_training_device(-1, indx, self.args.gpu_server_num)
        # local_max_acc = registry.get(f"client_{indx}_max_acc", 0.0)

        test_acc, test_loss = evaluation(test_loader, self._model,
                                         self.args.gpu, self.cuda)
        # if local_max_acc < test_acc:
        #     local_max_acc = test_acc
        #     registry.register(f"client_{indx}_max_acc", round(local_max_acc, 5))
        # if self.wandb:
        #     self.wandb.log({f"client_{indx}_GlobalTestLoss": test_loss,
        #                     f"client_{indx}_GlobalTestAcc": test_acc,
        #                     f"client_{indx}_MaxGlobalTestAcc": local_max_acc})
        return test_acc, test_loss


class MyScaleClientPassiveManager(ClientPassiveManager):
    def __init__(self, network, trainer, id_list, logger=None):
        super().__init__(network, trainer, logger)
        self.id_list = id_list

    def main_loop(self):
        """Actions to perform when receiving new message, including local training."""
        while True:
            sender_rank, message_code, payload = PackageProcessor.recv_package(src=0)
            if message_code == MessageCode.Exit:
                break
            elif message_code == MessageCode.ParameterUpdate:
                model_parameters = payload[0]

                _, message_code, payload = PackageProcessor.recv_package(src=0)
                # id_list = payload[0].tolist()
                _, message_code, payload = PackageProcessor.recv_package(src=0)
                round = payload[0].numpy()[0]

                # check the trainer type
                if self._trainer.type == SERIAL_TRAINER:
                    self.model_parameters_list = self._trainer.train(
                        model_parameters=model_parameters,
                        id_list=self.id_list,
                        aggregate=False,
                        round=round)
                elif self._trainer.type == ORDINARY_TRAINER:
                    self.model_parameters_list = self._trainer.train(
                        model_parameters=model_parameters)
                self.synchronize()
            else:
                raise ValueError("Invalid MessageCode {}. Please see MessageCode Enum".format(message_code))

    def synchronize(self):
        """Synchronize local model with server actively"""
        pack = Package(message_code=MessageCode.ParameterUpdate,
                       content=self.model_parameters_list)
        PackageProcessor.send_package(package=pack, dst=0)


class MyScaleSynchronousManager(ServerSynchronousManager):
    """ServerManager used in scale scenario."""

    def __init__(self, network, handler, logger=None):
        super().__init__(network, handler, logger)

    def activate_clients(self):
        """Use client id mapping: Coordinator.

        Here we use coordinator to find the rank client process with specific client_id.
        """
        clients_this_round = self._handler.sample_clients()
        rank_dict = self.coordinator.map_id_list(clients_this_round)

        self._LOGGER.info("Client Activation Procedure")
        for rank, values in rank_dict.items():
            self._LOGGER.info("rank {}, client ids {}".format(rank, values))

            # Send parameters
            param_pack = Package(message_code=MessageCode.ParameterUpdate,
                                 content=self._handler.model_parameters)
            PackageProcessor.send_package(package=param_pack, dst=rank)

            # Send activate id list
            id_list = torch.Tensor(values).to(torch.int32)
            act_pack = Package(message_code=MessageCode.ParameterUpdate,
                               content=id_list)
            PackageProcessor.send_package(package=act_pack, dst=rank)

            # Send round
            round = torch.Tensor([self._handler.round]).to(torch.int32)
            round_pack = Package(message_code=MessageCode.ParameterUpdate,
                                 content=round)
            PackageProcessor.send_package(package=round_pack, dst=rank)

    def main_loop(self):
        while self._handler.stop_condition() is not True:
            activate = threading.Thread(target=self.activate_clients)
            activate.start()

            while True:
                sender, message_code, payload = PackageProcessor.recv_package()
                if message_code == MessageCode.ParameterUpdate:
                    for model_parameters in payload:
                        updated = self._handler.add_model(sender, model_parameters)

                    if updated:
                        break
                else:
                    raise Exception(
                        "Unexpected message code {}".format(message_code))
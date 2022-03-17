import random
import threading

import torch

from fedlab.utils.aggregator import Aggregators
from fedlab.core.server.handler import ParameterServerBackendHandler
from fedlab.utils.message_code import MessageCode
from fedlab.core.client.trainer import ClientTrainer
from fedlab.utils.serialization import SerializationTool
from fedlab.core.communicator.processor import Package, PackageProcessor
from fedlab.core.client.manager import ClientPassiveManager
from fedlab.core.client import SERIAL_TRAINER, ORDINARY_TRAINER
from fedlab.core.server import ServerSynchronousManager

from training.utils.register import registry


class TLMSyncParameterServerHandler(ParameterServerBackendHandler):

    def __init__(self,
                 model,
                 global_test_data,
                 global_round=5,
                 cuda=True,
                 sample_ratio=1.0,
                 logger=None,
                 gpu=None,
                 trainer=None):
        super(TLMSyncParameterServerHandler, self).__init__(model, cuda)

        self.global_test_data = global_test_data
        self.gpu = gpu
        self.trainer = trainer
        self.args = trainer.args
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
        result, _, _ = self.trainer.eval_model(
            model=self._model,
            test_dl=self.global_test_data)
        test_acc, test_loss = result["acc"], result["eval_loss"]
        if self.test_global_max_acc < test_acc:
            self.test_global_max_acc = test_acc
        self._LOGGER.critical(f"{self.args.dataset}-{self.args.model_type} "
                              f"train with niid={self.args.niid}_lr={self.args.lr}_"
                              f"epoch={self.args.epochs}_seed={self.args.seed}_"
                              f"comm_round={self.args.comm_round}")
        self._LOGGER.critical(f"Server Testing "
                              f"Round: {self.round}, Current Acc: {round(test_acc, 3)}, "
                              f"Current Loss: {round(test_loss, 3)}, Max Acc: {round(self.test_global_max_acc, 3)}")
        if self.wandb:
            self.wandb.log({"server_GlobalTestLoss": round(test_loss, 3),
                            "server_GlobalTestAcc": round(test_acc, 3),
                            "server_GlobalMaxAcc": round(self.test_global_max_acc, 3)})


#### Scale Training
class TLMSerialTrainer(ClientTrainer):
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
        raise NotImplementedError()

    def _get_dataloader(self, dataset, client_id: int):
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


class TLMSubsetSerialTrainer(TLMSerialTrainer):
    def __init__(self,
                 model,
                 train_dataset,
                 test_dataset,
                 data_slices,
                 aggregator=None,
                 logger=None,
                 cuda=True,
                 args=None,
                 embedder=None,
                 trainer=None) -> None:

        super(TLMSubsetSerialTrainer, self).__init__(model=model,
            client_num=len(data_slices),
            cuda=cuda,
            aggregator=aggregator,
            logger=logger)

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.data_slices = data_slices  # [0, client_num)
        self.args = args
        self.embedder = embedder
        self.trainer = trainer
        self.wandb = registry.get("wandb")

    def _get_dataloader(self, dataset, client_id: int):
        if isinstance(dataset, dict):
            train_loader = dataset[client_id]
        else:
            train_loader = dataset
        return train_loader

    def _train_alone(self, model_parameters, train_loader, indx, round=None):
        SerializationTool.deserialize_model(self._model, model_parameters)
        # train_model(self, model, train_dl, test_dl):
        self.trainer.train_model(model=self._model, train_dl=train_loader)
        self._LOGGER.info(f"Client id {indx} train procedure is finished")
        return self.model_parameters

    def _test_alone(self, test_loader, indx):
        # gpu = init_training_device(-1, indx, self.args.gpu_server_num)
        local_max_acc = registry.get(f"client_{indx}_max_acc", 0.0)

        result, _, _ = self.trainer.eval_model(model=self._model,
            test_dl=test_loader)
        test_acc, test_loss = result["acc"], result["eval_loss"]
        if local_max_acc < test_acc:
            local_max_acc = test_acc
            registry.register(f"client_{indx}_max_acc", round(local_max_acc, 5))
        if self.wandb:
            self.wandb.log({f"client_{indx}_GlobalTestLoss": test_loss,
                            f"client_{indx}_GlobalTestAcc": test_acc,
                            f"client_{indx}_MaxGlobalTestAcc": local_max_acc})
        return test_acc, test_loss, local_max_acc


class TLMScaleClientPassiveManager(ClientPassiveManager):
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


class TLMScaleSynchronousManager(ServerSynchronousManager):
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

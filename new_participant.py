import rich
from rich.progress import track
import torch
import utils
from copy import deepcopy
from typing import Tuple, Union
from collections import OrderedDict
from data.new_data_utils import get_dataloader, n_get_dataloader
from fedlab.utils.serialization import SerializationTool


class User:
    def __init__(
        self,
        user_id: int,
        user_role: str,
        lr: float,
        alpha: float,
        beta: float,
        global_model: torch.nn.Module,
        criterion: Union[torch.nn.CrossEntropyLoss, torch.nn.MSELoss],
        batch_size: int,
        dataset: str,
        local_epochs: int,
        valset_ratio: float,
        logger: rich.console.Console,
        gpu: int,
        gpu_option: bool,
        data_type: str,
        n_class: int,
        optm: str,
    ):
        if gpu and torch.cuda.is_available():
            if gpu_option:
                gpu_card = input("Enter the gpu card number you want to use (Enter x if wanna default):")
                if gpu_card != "x":
                    self.device = torch.device(f"cuda:{int(gpu_card)}")
                else: self.device = torch.device("cuda")
            else: self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.logger = logger
        self.n_class = n_class
        self.local_epochs = local_epochs
        self.criterion = criterion
        self.id = user_id
        self.role = user_role
        self.model = deepcopy(global_model)
        self.lr = lr
        self.alpha = alpha
        self.beta = beta
        self.optm = optm
        print(f'n_per-> dataset: {dataset}')
        if dataset == "nmnist":
            self.trainloader, self.valloader = n_get_dataloader(
            dataset, user_id, data_type, n_class, batch_size, valset_ratio
            )
        else:            
            self.trainloader, self.valloader = get_dataloader(
                dataset, user_id, n_class, batch_size, valset_ratio
            )
        self.iter_trainloader = iter(self.trainloader)

    def get_data_batch(self):
        try:
            x, y = next(self.iter_trainloader)
        except StopIteration:
            self.iter_trainloader = iter(self.trainloader)
            x, y = next(self.iter_trainloader)

        return x.to(self.device), y.to(self.device)

    def train(
        self,
        global_model: torch.nn.Module,
        hessian_free=False,
        eval_while_training=False, training_stats=None,
    ):
        loss_before, loss_after, acc_before, acc_after = None, None, None, None
        self.model.load_state_dict(global_model.state_dict())
        if eval_while_training:
            loss_before, acc_before, pred, target = utils.eval(
                self.model, self.valloader, self.criterion, self.device
            )

        self.logger.log(
                "User [{:d}] [red]role: {:s}  [blue]hessian: {:b}".format(
                    self.id,
                    str(self.role),
                    hessian_free,
                )
            )
        
        training_stats = self.meta_train(hessian_free, training_stats)
        
        if eval_while_training:
            loss_after, acc_after, pred, target = utils.eval(
                self.model, self.valloader, self.criterion, self.device
            )
            self.logger.log(
                "client [{}] [red]loss: {:.4f} -> {:.4f}   [blue]acc: {:.2f}% -> {:.2f}%".format(
                    self.id,
                    loss_before,
                    loss_after,
                    acc_before * 100.0,
                    acc_after * 100.0,
                )
            )
        return SerializationTool.serialize_model(self.model), training_stats

    def meta_train(self, hessian_free=False, training_stats= None,):
        self.logger.log("Meta Training at user: {} .................%" .format(self.id))
        if hessian_free:  # Per-FedAvg(HF)
            for _ in range(self.local_epochs):
                temp_model = deepcopy(self.model)
                data_batch_1 = self.get_data_batch()
                grads, training_stats = self.compute_grad(temp_model, data_batch_1, training_stats)
                for param, grad in zip(temp_model.parameters(), grads):
                    param.data.sub_(self.alpha * grad)

                data_batch_2 = self.get_data_batch()
                grads_1st, training_stats = self.compute_grad(temp_model, data_batch_2, training_stats)

                data_batch_3 = self.get_data_batch()

                grads_2nd, training_stats = self.compute_grad(
                    self.model, data_batch_3, training_stats, v=grads_1st, second_order_grads=True
                )
                # NOTE: Go check https://github.com/KarhouTam/Per-FedAvg/issues/2 if you confuse about the model update.
                for param, grad1, grad2 in zip(
                    self.model.parameters(), grads_1st, grads_2nd
                ):
                    param.data.sub_(self.beta * grad1 - self.beta * self.alpha * grad2)

        else:  # Per-FedAvg(FO)
            for _ in range(self.local_epochs):
                # ========================== FedAvg ==========================
                # NOTE: You can uncomment those codes for running FedAvg.
                #       When you're trying to run FedAvg, comment other codes in this branch.

                # data_batch = self.get_data_batch()
                # grads = self.compute_grad(self.model, data_batch)
                # for param, grad in zip(self.model.parameters(), grads):
                #     param.data.sub_(self.beta * grad)

                # ============================================================

                temp_model = deepcopy(self.model)
                data_batch_1 = self.get_data_batch()
                grads, training_stats = self.compute_grad(temp_model, data_batch_1, training_stats)

                for param, grad in zip(temp_model.parameters(), grads):
                    param.data.sub_(self.alpha * grad)

                data_batch_2 = self.get_data_batch()
                grads, training_stats = self.compute_grad(temp_model, data_batch_2, training_stats)

                for param, grad in zip(self.model.parameters(), grads):
                    param.data.sub_(self.beta * grad)
        return training_stats

    def compute_grad(
        self,
        model: torch.nn.Module,
        data_batch: Tuple[torch.Tensor, torch.Tensor], training_stats,
        v: Union[Tuple[torch.Tensor, ...], None] = None,
        second_order_grads=False,
    ):
        x, y = data_batch
        
        if second_order_grads:
            frz_model_params = deepcopy(model.state_dict())
            delta = 1e-3
            dummy_model_params_1 = OrderedDict()
            dummy_model_params_2 = OrderedDict()
            with torch.no_grad():
                for (layer_name, param), grad in zip(model.named_parameters(), v):
                    dummy_model_params_1.update({layer_name: param + delta * grad})
                    dummy_model_params_2.update({layer_name: param - delta * grad})

            model.load_state_dict(dummy_model_params_1, strict=False)
            logit_1 = model(x)
            loss_1 = self.criterion(logit_1, y)
            grads_1 = torch.autograd.grad(loss_1, model.parameters())

            model.load_state_dict(dummy_model_params_2, strict=False)
            logit_2 = model(x)
            loss_2 = self.criterion(logit_2, y)
            grads_2 = torch.autograd.grad(loss_2, model.parameters())

            model.load_state_dict(frz_model_params)
            # monitoring training stats
            for label in y:
                if label.item() not in training_stats['meta']:
                    training_stats['meta'][label.item()]=1
                else:
                    training_stats['meta'][label.item()] += 1
            grads = []
            with torch.no_grad():
                for g1, g2 in zip(grads_1, grads_2):
                    grads.append((g1 - g2) / (2 * delta))
            return grads, training_stats

        else:
            logit = model(x)
            loss = self.criterion(logit, y)
            grads = torch.autograd.grad(loss, model.parameters())
            # monitoring training stats
            for label in y:
                if label.item() not in training_stats['meta']:
                    training_stats['meta'][label.item()]=1
                else:
                    training_stats['meta'][label.item()] += 1
            return grads, training_stats
    def pers_params(self, local_model: torch.nn.Module, worker_epochs: int, training_stats):
        self.logger.log("Petsonalized Training at user: {}  ..................%" .format(self.id))
        SerializationTool.deserialize_model(self.model, local_model)
        # self.model.load_state_dict(local_model)
        loss_before, acc_before, pred, target = utils.eval(
            self.model, self.valloader, self.criterion, self.device
        )
                
        if self.optm == "SGD":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        elif self.optm =="Adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        else: optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        for _ in range(worker_epochs):
            x, y = self.get_data_batch()
            logit = self.model(x)            
            loss = self.criterion(logit, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # pred = torch.softmax(logit, -1).argmax(-1) # Testing
            # print(f"Predicted: {pred}, Target: {y}") #Print
            # monitoring training stats
            for label in y:
                if label.item() not in training_stats['pers']:
                    training_stats['pers'][label.item()]=1
                else:
                    training_stats['pers'][label.item()] += 1

        loss_after, acc_after, pred, target = utils.eval(
            self.model, self.valloader, self.criterion, self.device
        )
        self.logger.log(
            "client [{}] [red]loss: {:.4f} -> {:.4f}   [blue]acc: {:.2f}% -> {:.2f}%".format(
                self.id, loss_before, loss_after, acc_before * 100.0, acc_after * 100.0
            )
        )
        return {
            "model": SerializationTool.serialize_model(self.model),
            "loss_before": loss_before,
            "acc_before": acc_before,
            "loss_after": loss_after,
            "acc_after": acc_after,
            "training_stats": training_stats,
        }

    def pers_N_eval(self, global_model: torch.nn.Module, pers_epochs: int):
        self.model.load_state_dict(global_model.state_dict())

        loss_before, acc_before, pred, target = utils.eval(
            self.model, self.valloader, self.criterion, self.device
        )
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.alpha)
        for _ in range(pers_epochs):
            x, y = self.get_data_batch()
            logit = self.model(x)            
            loss = self.criterion(logit, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pred = torch.softmax(logit, -1).argmax(-1) # Testing
            # print(f'Logot: {logit}')
            print(f"Predicted: {pred}, Target: {y}") #Print
        loss_after, acc_after, pred, target = utils.eval(
            self.model, self.valloader, self.criterion, self.device
        )
        self.logger.log(
            "client [{}] [red]loss: {:.4f} -> {:.4f}   [blue]acc: {:.2f}% -> {:.2f}%".format(
                self.id, loss_before, loss_after, acc_before * 100.0, acc_after * 100.0
            )
        )
        return {
            "loss_before": loss_before,
            "acc_before": acc_before,
            "loss_after": loss_after,
            "acc_after": acc_after,
        }
    
    def client_update(self,
        lr: float,
        alpha: float,
        beta: float,
        global_model: torch.nn.Module,
        criterion: Union[torch.nn.CrossEntropyLoss, torch.nn.MSELoss],
        batch_size: int,
        dataset: str,
        local_epochs: int,
        valset_ratio: float,
        logger: rich.console.Console,
        gpu: int,
    ):
        if gpu and torch.cuda.is_available():
            gpu_card = input("Enter the gpu card number you want to use (Enter x if wanna default):")
            if gpu_card != "x":
                self.device = torch.device(f"cuda:{int(gpu_card)}")
            else: self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.logger = logger

        self.local_epochs = local_epochs
        self.criterion = criterion
        self.model = deepcopy(global_model)
        self.lr = lr
        self.alpha = alpha
        self.beta = beta
        self.trainloader, self.valloader = get_dataloader(
            dataset, self.id, batch_size, valset_ratio
        )
        self.iter_trainloader = iter(self.trainloader)

    def test_other_model(self, model: torch.nn.Module, show: bool, record_stats):
        loss, acc, pred, target = utils.eval(
                model, self.valloader, self.criterion, self.device, show = show,
            )
        
        if show:
            print(f"Predicted: {pred}, Target: {target}") #Print
            # print(record_stats)
            if "pred" not in record_stats:
                record_stats["pred"] = {}
                record_stats["target"] = {}
                record_stats["match"] = {}
            for x, y in zip(pred, target):
                X = x.item()
                Y = y.item()
                if X not in record_stats["pred"]:
                    print(f'predicted new class {X}')
                    record_stats["pred"][X] = 0
                record_stats["pred"][X] += 1
                if Y not in record_stats["target"]:
                    print(f'predicted new class {Y}')
                    record_stats["target"][Y] = 0
                record_stats["target"][Y] += 1
                if X==Y :
                    if Y not in record_stats["match"]:
                        print(f'new matced: {X}, {Y}')
                        record_stats["match"][Y] = 0
                    record_stats["match"][Y] += 1



        return loss, acc, record_stats



        

import sys
sys.path.append("data")
from new_participant import User
from torch import nn
import torch
from fedlab.utils.serialization import SerializationTool
from fedlab.utils.aggregator import Aggregators
from copy import deepcopy
from rich.console import Console
from rich.progress import track
import random
import numpy as np

from data.new_data_utils import get_client_id_indices
from data.new_data_utils import get_dataset_stat
from utils import get_args, fix_random_seed
from model import get_model



dataset = 'nmnist'
a = list(range(200))
b = random.sample(a, 160)
clients_4_training, clients_4_eval, client_num_in_total = b, [x for x in a if x not in b], 200
args = get_args()
logger = Console(record=args.log)
logger.log(f"Arguments:", dict(args._get_kwargs()))
device = "cuda" if torch.cuda.is_available() else "cpu"
global_model = get_model(args.dataset, device)
client_num_in_total = 200
participants = [
        User(
            user_id=user_id,
            user_role= None,
            lr=args.lr,
            alpha=args.alpha,
            beta=args.beta,
            global_model=global_model,
            criterion=torch.nn.CrossEntropyLoss(),
            batch_size=args.batch_size,
            dataset=args.dataset,
            local_epochs=args.local_epochs,
            valset_ratio=args.valset_ratio,
            logger=logger,
            gpu=args.gpu,
            gpu_option=False,
            data_type=args.data_type,
            n_class=args.n_class,
            optm=args.optm,
        )
        for user_id in range(client_num_in_total)
    ]

training_stats = {'meta': {}, 'pers': {}}
logger.log("=" * 20, "TRAINING", "=" * 20, style="bold red")
for round in track(
        range(args.global_epochs), "Training...", console=logger, disable=args.log
    ):
    params_cache = []
    selected_learners = random.sample(clients_4_training, args.client_num_per_round)
    for user_id in selected_learners:
        temp_model = deepcopy(global_model)
        personalized_stats = participants[user_id].pers_params(local_model= SerializationTool.serialize_model(global_model), worker_epochs= args.worker_epochs, training_stats=training_stats)
        params_cache.append([personalized_stats['model'], personalized_stats['acc_after']])
        global_model = deepcopy(temp_model)
        training_stats = personalized_stats['training_stats']
    local_params = [x[0] for x in params_cache]
    local_acc = [int((x[1].item()+0.1)*100) for x in params_cache]
    logger.log("->"*5, 'AGGREGATING', '<-'*5, style='bold blue' )
    print(f"Local Accuracy: {local_acc}")
    aggregated_model_params = Aggregators.fedavg_aggregate(local_params, local_acc)
    SerializationTool.deserialize_model(global_model, aggregated_model_params)

logger.log("=" * 20, "EVALUATING", "=" * 20, style="bold red")
record_stats = {}
for client in clients_4_eval:
    loss, acc, record_stats = participants[client].test_other_model(model=global_model, show = True, record_stats= record_stats,)

    print(f' Testing model Trained for {args.global_epochs} epochs\n Training loss:{None}\n testing model for client id: {participants[client].id}, loss: {loss}, acc: {acc}')

print(record_stats)
record_stats = {"pred":{}, "target":{}, "match":{}}
logger.log("=" * 20, "EVALUATING FOR ALL", "=" * 20, style="bold red")
total_acc = []
zero_acc = 0
for client in range(200):
    loss, acc, record_stats = participants[client].test_other_model(model=global_model, show = True, record_stats= record_stats,)
    ACC = acc.item()
    total_acc.append(ACC)
    if ACC <=0:
        zero_acc += 1
    print(f' Testing model Trained for {args.global_epochs} epochs\n Training loss:{None}\n testing model for client id: {participants[client].id}, loss: {loss}, acc: {acc}')
print(record_stats)
mean_acc = np.mean(total_acc)
print(f'Total mean accuracy: {mean_acc}, zero obtain: {zero_acc} \n Training Stats: {training_stats}')
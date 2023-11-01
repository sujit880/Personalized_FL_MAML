import sys

sys.path.append("data")

import torch
import numpy as np
import random
import os
from fedlab.utils.serialization import SerializationTool
from fedlab.utils.aggregator import Aggregators
from rich.console import Console
from rich.progress import track
from utils import get_args, fix_random_seed
from model import get_model
from participant import User
from data.data_utils import get_client_id_indices

ROLE = ['learner', 'worker']

if __name__ == "__main__":
    args = get_args()
    # print(f'args:-\n> {args} \n !') 
    fix_random_seed(args.seed)
    if os.path.isdir("./log") == False:
        os.mkdir("./log")
    if args.gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    global_model = get_model(args.dataset, device)
    logger = Console(record=args.log)
    logger.log(f"Arguments:", dict(args._get_kwargs()))
    clients_4_training, clients_4_eval, client_num_in_total = get_client_id_indices(
        args.dataset
    )
    print(f"Total Number of Clients: {client_num_in_total}")
    # init clients
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
        )
        for user_id in range(client_num_in_total)
    ]
    participation_stats = {}
    for user in participants:
        participation_stats[user.id] = {'learner': {'round': [], 'loss':[],'acc':[]}, 'worker': {'round': [], 'loss':[],'acc':[]} }
    # training
    logger.log("=" * 20, "TRAINING", "=" * 20, style="bold red")
    for round in track(
        range(args.global_epochs), "Training...", console=logger, disable=args.log
    ):
        # select clients
        selected_learners = random.sample(clients_4_training, args.client_num_per_round)

        model_params_cache = {}
        # client local training
        for user_id in selected_learners:
            participants[user_id].role = ROLE[0]
            serialized_model_params = participants[user_id].train(
                global_model=global_model,
                hessian_free=args.hf,
                eval_while_training=args.eval_while_training,
            )
            model_params_cache[user_id] =serialized_model_params
        # select workers
        remaining_users = [i for i in clients_4_training if i not in selected_learners]
        required_workers = 2*len(selected_learners) if 2*len(selected_learners)<= len(remaining_users) else len(selected_learners)
        selected_workers = random.sample(clients_4_training, required_workers)
        assigned_wroker_per_task = int(len(selected_workers)/len(selected_learners))
        aggr_params_cache = {}
        used_workers = []
        for learner_id in model_params_cache.keys():            
            participants[user_id].role = ROLE[1] #set role
            remaining_workers = [i for i in selected_workers if i not in used_workers]
            selected_task_pers = random.sample(remaining_workers, assigned_wroker_per_task)
            used_workers = used_workers + selected_task_pers
            for worker_id in selected_task_pers:
                local_model = model_params_cache[learner_id]
                personalized_stats = participants[user_id].pers_params(local_model= local_model, worker_epochs= args.worker_epochs)
                if learner_id not in aggr_params_cache:
                    aggr_params_cache[learner_id] = [[personalized_stats['model'], personalized_stats['acc_after'], worker_id]]
                else: aggr_params_cache[learner_id].append([personalized_stats['model'], personalized_stats['acc_after']])
        # processing workers personalized parametrs
        local_aggr_params = []
        weights = []
        for learner_id in aggr_params_cache.keys():
            local_params = [x[0] for x in aggr_params_cache[learner_id]]
            local_acc = [x[1].cpu() for x in aggr_params_cache[learner_id]]
            agr_local_params =  Aggregators.fedavg_aggregate(local_params)
            local_aggr_params.append(agr_local_params)
            weights.append(np.mean(local_acc)+0.1)
        print(f'weights: {weights}')
        # aggregate model parameters
        aggregated_model_params = Aggregators.fedavg_aggregate(local_aggr_params, weights)
        SerializationTool.deserialize_model(global_model, aggregated_model_params)
        logger.log("=" * 60)
    # eval
    pers_epochs = args.local_epochs if args.pers_epochs == -1 else args.pers_epochs
    logger.log("=" * 20, "EVALUATION", "=" * 20, style="bold blue")
    loss_before = []
    loss_after = []
    acc_before = []
    acc_after = []
    clients_4_eval = [i for i in range(0,199)]
    # eval_epochs = 30
    for client_id in track(
        clients_4_eval, "Evaluating...", console=logger, disable=args.log
    ):
        stats = participants[client_id].pers_N_eval(
            global_model=global_model, pers_epochs=args.eval_epochs,
        )
        loss_before.append(stats["loss_before"])
        loss_after.append(stats["loss_after"])
        acc_before.append(stats["acc_before"])
        acc_after.append(stats["acc_after"])

    logger.log("=" * 20, "RESULTS", "=" * 20, style="bold green")
    logger.log(f"loss_before_pers: {(sum(loss_before) / len(loss_before)):.4f}")
    logger.log(f"acc_before_pers: {(sum(acc_before) * 100.0 / len(acc_before)):.2f}%")
    logger.log(f"loss_after_pers: {(sum(loss_after) / len(loss_after)):.4f}")
    logger.log(f"acc_after_pers: {(sum(acc_after) * 100.0 / len(acc_after)):.2f}%")

    if args.log:
        algo = "HF" if args.hf else "FO"
        logger.save_html(
            f"./log/{args.dataset}_{args.client_num_per_round}_{args.global_epochs}_{pers_epochs}_{algo}.html"
        )


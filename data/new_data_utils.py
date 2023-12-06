import pickle
import os
from torch.utils.data import random_split, DataLoader
from dataset import MNISTDataset, CIFARDataset, N_MNISTDataset
from path import Path

DATASET_DICT = {
    "mnist": MNISTDataset,
    "cifar": CIFARDataset,
    "nmnist": N_MNISTDataset,
}
CURRENT_DIR = f'/home/sujit_2021cs35/Github/Personalized_FL_MAML/build_dataset'


def get_dataloader(dataset: str, client_id: int, num_classes, batch_size=20, valset_ratio=0.1):
    pickles_dir = f'{CURRENT_DIR}/{dataset}/{num_classes}'
    print(f'n_data_utils-> dataset: {dataset}, DATASET_DICT: {DATASET_DICT[dataset]}')
    print(f'pickles_dir: {pickles_dir}')
    if os.path.isdir(pickles_dir) is False:
        raise RuntimeError("Please preprocess and create pickles first.")

    with open(f'{pickles_dir}/{client_id}.pkl', "rb") as f:
        client_dataset: DATASET_DICT[dataset] = pickle.load(f)

    val_num_samples = int(valset_ratio * len(client_dataset))
    train_num_samples = len(client_dataset) - val_num_samples

    trainset, valset = random_split(
        client_dataset, [train_num_samples, val_num_samples]
    )
    trainloader = DataLoader(trainset, batch_size, drop_last=True)
    valloader = DataLoader(valset, batch_size)

    return trainloader, valloader

def n_get_dataloader(dataset: str, client_id: int, data_type: str, n_class: int, batch_size=20, valset_ratio=0.1):
    pickles_dir = f'./build_dataset/nmnist/{data_type}/{n_class}'
    if os.path.isdir(pickles_dir) is False:
        raise RuntimeError("Please preprocess and create pickles first.")

    with open(f'{pickles_dir}/{client_id}.pkl', "rb") as f:
        client_dataset: DATASET_DICT[dataset] = pickle.load(f)

    val_num_samples = int(valset_ratio * len(client_dataset))
    train_num_samples = len(client_dataset) - val_num_samples

    trainset, valset = random_split(
        client_dataset, [train_num_samples, val_num_samples]
    )
    trainloader = DataLoader(trainset, batch_size, drop_last=True)
    valloader = DataLoader(valset, batch_size)

    return trainloader, valloader
 

def get_client_id_indices(dataset):
    print(f'Dataset Dir: {CURRENT_DIR}')
    dataset_pickles_path = CURRENT_DIR / dataset / "pickles"
    with open(dataset_pickles_path / "seperation.pkl", "rb") as f:
        seperation = pickle.load(f)
    return (seperation["train"], seperation["test"], seperation["total"])


def get_dataset_stat(dataset, data_type: str, class_type: int):
    #calculating datasets stat
    pickles_dir = f'{CURRENT_DIR}/{dataset}/{data_type}/{class_type}'
    DATASET_DICT = {
        "mnist": MNISTDataset,
        "cifar": CIFARDataset,
        "nmnist": N_MNISTDataset,
    }
    dataset_stats = {}
    for i in range(200):
        with open(f'{pickles_dir}/{i}.pkl', "rb") as f:
            client_dataset: DATASET_DICT[dataset] = pickle.load(f)
            if i not in dataset_stats:
                dataset_stats[i]={}
            for x in client_dataset.targets:
                if x.item() not in dataset_stats[i]:
                    dataset_stats[i][x.item()] = 0
                dataset_stats[i][x.item()] += 1
    return dataset_stats


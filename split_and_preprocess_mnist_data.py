import pandas as pd
import torch
import numpy as np
import random
import os
import torch
import pickle
from data.dataset import N_MNISTDataset
from torchvision.io import read_image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--data_type", type=str, default='non_iid')
    parser.add_argument("--number_of_classes", type=int, default=10)
    parser.add_argument("--total_classes", type=int, default=10)
    parser.add_argument("--number_of_dataset", type=int, default=200)
    parser.add_argument("--sub_dataset_size", type=int, default=500)
    parser.add_argument("--mean", type=float, default=0.5)
    parser.add_argument("--std", type=float, default=0.5)
    return parser.parse_args()

# Normalize array

def normalize_array(arr):
    if len(arr) <= 1:
        return arr
    else: return arr / np.sum(arr)

train_dict = {}
# function to load data from csv
def create_mnist_data_dict(data):
    data_dict = {}
    for index, row in  data.iterrows():
        if row['label'] not in data_dict:
            data_dict[row['label']] = []
        img = []
        for i in range(784):
            img.append(row[f'pixel{i}'])
        image = torch.tensor(img)
        image = torch.reshape(image, (28, 28))
        # print(image)
        # break
        data_dict[row['label']].append((image.numpy(), row['label']))
    return data_dict

def create_sub_dataset(dataset, minimum_number_of_class, total_classes, number_of_dataset, sub_dataset_size, data_type):
    created_dataset_dict = {}
    data_classes = list(range(total_classes))
    for i in range (number_of_dataset):
        sub_dataset_dict = {}
        ratios = None
        selected_classes = random.sample(data_classes, minimum_number_of_class)
        if data_type != 'iid':
            ratios = np.random.dirichlet(np.ones(len(selected_classes)))
        else:
            ratios = np.array([1/len(selected_classes) for _ in range(len(selected_classes))]) # IID Data
        class_sample_size = (ratios * sub_dataset_size).astype(int)
        for data_class, sample_size in zip(selected_classes, class_sample_size):
            sub_dataset_dict[data_class] = random.sample(range(len(dataset[data_class])), sample_size)
        created_dataset_dict[i] = sub_dataset_dict
    return created_dataset_dict
    
if __name__ == "__main__":
    args = get_args()
    print(f'Argements: {args}')

    total_classes = args.total_classes
    number_of_dataset = args.number_of_dataset
    sub_dataset_size = args.sub_dataset_size
    data_type = args.data_type

    train_filename = "/home/sujit_2021cs35/Github/Personalized_FL/dataset/mnist/train.csv"
    test_filename = "/home/sujit_2021cs35/Github/Personalized_FL/dataset/mnist/test.csv"

    data_train =pd.read_csv(train_filename)
    data_test = pd.read_csv(test_filename)



    train_dict = create_mnist_data_dict(data_train)   

    # function to split and preprocess the dataset
    MEAN = args.mean
    STD = args.std
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


    for number_of_classes in range(1,11):
        sub_dataset_dict = create_sub_dataset(train_dict,number_of_classes,total_classes,number_of_dataset,sub_dataset_size,data_type)  #'iid' for equal sample size

        for subset, sub_dict in sub_dataset_dict.items():
            total_samples = sum(len(indices) for indices in sub_dict.values())
            print(f'Id: {subset}, Lengths: {[len(sub_dict[x]) for x in sub_dict]}, Total: {total_samples}')
            
            data = [torch.tensor(train_dict[class_][index][0], dtype=torch.uint8) for class_, indices in sub_dict.items() for index in indices]
            labels = [train_dict[class_][index][1] for class_, indices in sub_dict.items() for index in indices]
            data = np.array(data)
            
            # print(f'Length: {len(data)}, Data: {data}')
            # Create an instance of your custom MNISTDataset
            mnist_dataset = N_MNISTDataset(data=data, targets=labels, transform=transform)
            # break
            directory_path = f'./build_dataset/nmnist/{data_type}/{number_of_classes}'
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)
            pickle_file_path = f"{directory_path}/{subset}.pkl"

            with open(pickle_file_path, 'wb') as file:
                pickle.dump(mnist_dataset, file)
   
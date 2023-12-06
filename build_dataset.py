import os
import pandas as pd
import numpy as np
import pickle
import random
import torch
from torchvision import transforms
from torch.utils.data import Dataset

# Define the MNISTDataset class (as in your original code)
class MNISTDataset(Dataset):
    def __init__(
        self,
        subset=None,
        data=None,
        targets=None,
        transform=None,
        target_transform=None,
    ) -> None:
        self.transform = transform
        self.target_transform = target_transform
        if (data is not None) and (targets is not None):
            self.data = data.unsqueeze(1)
            self.targets = targets
        elif subset is not None:
            self.data = torch.stack(
                list(
                    map(
                        lambda tup: tup[0]
                        if isinstance(tup[0], torch.Tensor)
                        else torch.tensor(tup[0]),
                        subset,
                    )
                )
            )
            self.targets = torch.stack(
                list(
                    map(
                        lambda tup: tup[1]
                        if isinstance(tup[1], torch.Tensor)
                        else torch.tensor(tup[1]),
                        subset,
                    )
                )
            )
        else:
            raise ValueError(
                "Data Format: subset: Tuple(data: Tensor / Image / np.ndarray, targets: Tensor) OR data: List[Tensor]  targets: List[Tensor]"
            )

    def __getitem__(self, index):
        data, targets = self.data[index], self.targets[index]

        if self.transform is not None:
            data = self.transform(self.data[index])

        if self.target_transform is not None:
            targets = self.target_transform(self.targets[index])

        return data, targets

    def __len__(self):
        return len(self.targets)
    
# Read train data
train_filename = "dataset/mnist/train.csv"
data_train = pd.read_csv(train_filename)

# Function to create data dictionary
def create_mnist_data_dict(data):
    data_dict = {}
    for _, row in data.iterrows():
        label = row['label']
        if label not in data_dict:
            data_dict[label] = []
        img = [row[f'pixel{i}'] for i in range(784)]
        data_dict[label].append((img, label))
    return data_dict

train_dict = create_mnist_data_dict(data_train)

# Define your transformation
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])

# Define the create_sub_dataset function
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

# Create sub-datasets
sub_dataset_dict = create_sub_dataset(train_dict, 5, 10, 200, 500, 'iid')

for subset, sub_dict in sub_dataset_dict.items():
    total_samples = sum(len(indices) for indices in sub_dict.values())
    print(f'Id: {subset}, Lengths: {[len(sub_dict[x]) for x in sub_dict]}, Total: {total_samples}')

    data = [train_dict[class_][index][0] for class_, indices in sub_dict.items() for index in indices]
    labels = [train_dict[class_][index][1] for class_, indices in sub_dict.items() for index in indices]

    # Create an instance of your custom MNISTDataset
    mnist_dataset = MNISTDataset(data=torch.tensor(data), targets=torch.tensor(labels), transform=transform)

    # Define the directory path and pickle file path
    directory_path = f'build_dataset/mnist/{5}'
    os.makedirs(directory_path, exist_ok=True)
    pickle_file_path = f"{directory_path}/{subset}.pkl"

    # Save the dataset to a pickle file
    with open(pickle_file_path, 'wb') as file:
        pickle.dump(mnist_dataset, file)

# Loading a pickle file
directory_path = f'build_dataset/mnist/{5}'
pickle_file_path = f"{directory_path}/mnist/{5}/0.pkl"
loaded_dataset = None

if os.path.exists(pickle_file_path):
    with open(pickle_file_path, 'rb') as file:
        loaded_dataset = pickle.load(file)

print(loaded_dataset)

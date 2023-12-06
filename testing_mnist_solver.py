import sys
sys.path.append("data")
from participant import User
from torch import nn
import torch
clients = [User(
            user_id=i,
            user_role= None,
            lr=0.001,
            alpha=0.0001,
            beta=0.0001,
            global_model=None,
            criterion=nn.CrossEntropyLoss(),
            batch_size=128,
            dataset='mnist',
            local_epochs=128,
            valset_ratio=0.2,
            logger=None,
            gpu=1,
            gpu_option=False,
        ) for i in range(200)]
activation = nn.ELU
features = 28 * 28   # number of flattened array
classes = 10 
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(features, 128),
    activation(),
    nn.Linear(128, 128),
    activation(),
    nn.Linear(128, classes),
    # don't need softmax here!
)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
optimizer = torch.optim.Adam(model.parameters())

# Load saved model
PATH = '/home/sujit_2021cs35/Github/Personalized_FL/model/mnist/'
checkpoint = torch.load(f'{PATH}model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
t_loss = checkpoint['loss']
record_stats = {}
for client in clients:
    loss, acc, record_stats = client.test_other_model(model=model, show = True, record_stats = record_stats,)

    print(f' Testing model Trained for {epoch} epochs\n Training loss:{t_loss}\n testing model for client id: {client.id}, loss: {loss}, acc: {acc}')

print(f'Stats: {record_stats}')

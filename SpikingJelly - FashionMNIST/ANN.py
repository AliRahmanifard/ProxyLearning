from abc import ABC
import torch
import torch.nn as nn
import numpy as np
import copy
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
from tqdm import tqdm
from matplotlib import pyplot as plt


class ArtificialNeuralNetwork(nn.Module, ABC):
    def __init__(self):
        super().__init__()

        self.fullyConnected = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 1000, bias=True),
            nn.Sigmoid(),
            nn.Linear(1000, 10, bias=True),
        )

    def forward(self, x):
        output = self.fullyConnected(x)
        return output


torch.manual_seed(1)
np.random.seed(1)

device = "cpu"  # "cpu" or "cuda:0"
dataset_dir = "./"  # root directory for saving dataset
batch_size = 300  # batch size
learning_rate = 1e-3  # learning rate
train_epoch = 100  # training epochs

train_dataset = FashionMNIST(root=dataset_dir, train=True, transform=ToTensor(), download=False)
test_dataset = FashionMNIST(root=dataset_dir, train=False, transform=ToTensor(), download=False)

train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
test_data_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

num_of_classes = len(train_dataset.classes)
num_of_train_data = len(train_dataset.data)

ann = ArtificialNeuralNetwork()

ann.to(device)

optimizer_ann = torch.optim.Adam(ann.parameters(), lr=learning_rate, betas=(0.8, 0.99), eps=1e-08, weight_decay=1e-06)

epoch_loss_array = np.array([])
ann_train_acc_array = np.array([])
ann_test_acc_array = np.array([])
ann_test_acc_max = 0
ann_test_acc_max_epoch = 0
ann_best_model = None

for epoch in range(train_epoch):
    ann.train()
    ann_trained_num = 0
    ann_trained_correct = 0
    epoch_loss = np.array([])

    for img, label in tqdm(train_data_loader):
        img = img.to(device)
        label = label.to(device)
        label_one_hot = (nn.functional.one_hot(label, num_of_classes).float()).to(device)

        optimizer_ann.zero_grad()

        outputs = ann(img)
        predicted_ann = torch.max(outputs.data, 1)[1]
        ann_trained_num += label.numel()
        ann_trained_correct += (predicted_ann == label).sum().item()

        loss = nn.functional.mse_loss(outputs, label_one_hot)
        epoch_loss = np.append(epoch_loss, loss.data.item())
        loss.backward()
        optimizer_ann.step()

    ann.eval()
    with torch.no_grad():
        ann_tested_num = 0
        ann_tested_correct = 0
        for img, label in tqdm(test_data_loader):
            img = img.to(device)
            label = label.to(device)
            outputs = ann(img)
            predicted_ann = torch.max(outputs.data, 1)[1]
            ann_tested_correct += (predicted_ann == label).sum().item()
            ann_tested_num += label.numel()

    ann_train_acc = 100 * ann_trained_correct / ann_trained_num
    ann_test_acc = 100 * ann_tested_correct / ann_tested_num
    epoch_loss_array = np.append(epoch_loss_array, np.mean(epoch_loss))
    ann_train_acc_array = np.append(ann_train_acc_array, ann_train_acc)
    ann_test_acc_array = np.append(ann_test_acc_array, ann_test_acc)
    if ann_test_acc > ann_test_acc_max:
        ann_test_acc_max = ann_test_acc
        ann_test_acc_max_epoch = epoch + 1
        ann_best_model = copy.deepcopy(ann)

    print(f'Epoch {epoch+1} -> Train: {ann_train_acc:.2f}          Test: {ann_test_acc:.2f}          Loss: {np.mean(epoch_loss):.5f}')

print(f'Learning is Done! Best test accuracy was {ann_test_acc_max:.2f} in epoch {ann_test_acc_max_epoch}')

torch.save(ann_best_model, f'ANN - {ann_test_acc_max:.2f}.pt')

plt.plot(epoch_loss_array, label='Train loss')
plt.title('ANN Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('ANN loss.png', dpi=200)
plt.clf()
plt.close()

plt.plot(ann_train_acc_array, label='Train')
plt.plot(ann_test_acc_array, label='Test')
plt.title('ANN Accuracies')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('ANN Accuracies.png', dpi=200)
plt.clf()
plt.close()

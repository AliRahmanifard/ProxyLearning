from abc import ABC
import torch
import torch.nn as nn
import numpy as np
import copy
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from tqdm import tqdm
from spikingjelly.clock_driven import surrogate, functional
from spikingjelly.clock_driven.neuron import BaseNode
from matplotlib import pyplot as plt


class OneSpikeIFNode(BaseNode, ABC):
    def __init__(self, v_threshold=1.0, v_reset=0.0, detach_reset=False, surrogate_function=surrogate.Sigmoid(), monitor_state=False):
        super().__init__()
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.detach_reset = detach_reset
        self.surrogate_function = surrogate_function
        self.monitor = monitor_state
        self.v = 0
        self.spike = None
        self.fire_mask = 0
        self.reset()

    def neuronal_charge(self, dv: torch.Tensor):
        self.v += dv

    def neuronal_fire(self):
        if self.monitor:
            if self.monitor['h'].__len__() == 0:
                if self.v_reset is None:
                    self.monitor['h'].append(self.v.data.cpu().numpy().copy())
                else:
                    self.monitor['h'].append(self.v.data.cpu().numpy().copy())
            else:
                self.monitor['h'].append(self.v.data.cpu().numpy().copy())
        self.spike = (self.v >= self.v_threshold) * (1.0 - self.fire_mask)
        self.fire_mask += self.spike
        if self.monitor:
            self.monitor['s'].append(self.spike.data.cpu().numpy().copy())

    def neuronal_reset(self):
        if self.detach_reset:
            spike = self.spike.detach()
        else:
            spike = self.spike
        if self.v_reset is None:
            self.v = self.v - spike * self.v_threshold
        else:
            self.v = (1 - spike) * self.v + spike * self.v_reset
        if self.monitor:
            self.monitor['v'].append(self.v.data.cpu().numpy().copy())

    def extra_repr(self):
        return f'v_threshold={self.v_threshold}, v_reset={self.v_reset}, detach_reset={self.detach_reset}'

    def set_monitor(self, monitor_state=True):
        if monitor_state:
            self.monitor = {'h': [], 'v': [], 's': []}
        else:
            self.monitor = False

    def forward(self, dv: torch.Tensor):
        self.neuronal_charge(dv)
        self.neuronal_fire()
        self.neuronal_reset()
        return self.spike

    def reset(self):
        if self.v_reset is None:
            self.v = 0.0
        else:
            self.v = self.v_reset
        self.spike = None
        self.fire_mask = 0
        if self.monitor:
            self.monitor = {'h': [], 'v': [], 's': []}


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


class SpikingNeuralNetwork(nn.Module, ABC):
    def __init__(self, simulationTime, neuronThreshold1, neuronThreshold2):
        super().__init__()
        self.simulationTime = simulationTime
        self.neuronThreshold1 = neuronThreshold1
        self.neuronThreshold2 = neuronThreshold2

        self.fullyConnected = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 1000, bias=True),
            OneSpikeIFNode(v_threshold=self.neuronThreshold1, monitor_state=True),
            nn.Linear(1000, 10, bias=True),
            OneSpikeIFNode(v_threshold=self.neuronThreshold2, monitor_state=True),
        )

    def forward(self, x):
        final_spikes = 0
        for t in range(self.simulationTime):
            final_spikes += (self.fullyConnected(x)) * (t + 1)

        final_spikes[final_spikes == 0] = self.simulationTime + 1
        final_spikes = ((self.simulationTime + 1) - final_spikes) / (self.simulationTime + 1)
        return final_spikes


torch.manual_seed(1)
np.random.seed(1)

device = "cpu"  # "cpu" or "cuda:0"
dataset_dir = "./"  # root directory for saving dataset
batch_size = 300  # batch size
learning_rate = 1e-3  # learning rate
threshold1 = 2.0
threshold2 = 4.0
simulation_time = 16
train_epoch = 100  # training epochs

train_dataset = MNIST(root=dataset_dir, train=True, transform=ToTensor(), download=False)
test_dataset = MNIST(root=dataset_dir, train=False, transform=ToTensor(), download=False)

train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
test_data_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

num_of_classes = len(train_dataset.classes)
num_of_train_data = len(train_dataset.data)

ann = ArtificialNeuralNetwork()
snn = SpikingNeuralNetwork(simulationTime=simulation_time, neuronThreshold1=threshold1, neuronThreshold2=threshold2)
ann.to(device)
snn.to(device)

if_layer1 = snn.fullyConnected[2]
if_layer2 = snn.fullyConnected[4]

params_ann = ann.named_parameters()
params_snn = snn.named_parameters()
dict_params_snn = dict(params_snn)
dict_params_ann = dict(params_ann)
names_snn = list(dict_params_snn.keys())
names_ann = list(dict_params_ann.keys())
for i in range(names_snn.__len__()):
    dict_params_snn[names_snn[i]].data = dict_params_ann[names_ann[i]].data

optimizer_ann = torch.optim.Adam(ann.parameters(), lr=learning_rate, betas=(0.8, 0.99), eps=1e-08, weight_decay=1e-06)

epoch_loss_array = np.array([])
snn_train_acc_array = np.array([])
snn_test_acc_array = np.array([])
snn_max_test_acc = 0
snn_max_test_acc_epoch = 0
snn_best_model = None
snn_layer1_spikes_count_per_input_array = np.array([])
snn_layer2_spikes_count_per_input_array = np.array([])

for epoch in range(train_epoch):
    ann.train()
    snn.train()
    ann_trained_num = 0
    ann_trained_correct = 0
    snn_trained_num = 0
    snn_trained_correct = 0
    epoch_loss = np.array([])
    snn_layer1_spikes_count = 0
    snn_layer2_spikes_count = 0

    for img, label in tqdm(train_data_loader, position=0):
        functional.reset_net(snn)
        img = img.to(device)
        label = label.to(device)
        label_one_hot = (nn.functional.one_hot(label, num_of_classes).float()).to(device)

        optimizer_ann.zero_grad()

        outputs = ann(img)
        predicted_ann = torch.max(outputs.data, 1)[1]
        ann_trained_num += label.numel()
        ann_trained_correct += (predicted_ann == label).sum().item()

        output_spikes = snn(img)
        snn_trained_num += label.numel()
        sorted_output_spikes = torch.sort(output_spikes, dim=-1, descending=True)
        snn_trained_correct += (sorted_output_spikes[1][:, 0] == label).int().sum().item()
        if_layer1_monitor_s = np.asarray(if_layer1.monitor['s'], dtype=int).T
        if_layer2_monitor_s = np.asarray(if_layer2.monitor['s'], dtype=int).T
        snn_layer1_spikes_count += np.sum(if_layer1_monitor_s)
        snn_layer2_spikes_count += np.sum(if_layer2_monitor_s)

        outputs.data.copy_(output_spikes)
        loss = nn.functional.mse_loss(outputs, label_one_hot)
        epoch_loss = np.append(epoch_loss, loss.data.item())
        loss.backward()
        optimizer_ann.step()

    ann.eval()
    snn.eval()
    with torch.no_grad():
        ann_tested_num = 0
        ann_tested_correct = 0
        snn_tested_num = 0
        snn_tested_correct = 0
        for img, label in test_data_loader:
            functional.reset_net(snn)
            img = img.to(device)
            label = label.to(device)
            outputs = ann(img)
            predicted_ann = torch.max(outputs.data, 1)[1]
            ann_tested_correct += (predicted_ann == label).sum().item()
            ann_tested_num += label.numel()

            output_spikes = snn(img)
            snn_tested_num += label.numel()
            sorted_output_spikes = torch.sort(output_spikes, dim=-1, descending=True)
            snn_tested_correct += (sorted_output_spikes[1][:, 0] == label).int().sum().item()

    ann_train_acc = 100 * ann_trained_correct / ann_trained_num
    snn_train_acc = 100 * snn_trained_correct / snn_trained_num
    ann_test_acc = 100 * ann_tested_correct / ann_tested_num
    snn_test_acc = 100 * snn_tested_correct / snn_tested_num
    epoch_loss_array = np.append(epoch_loss_array, np.mean(epoch_loss))
    snn_train_acc_array = np.append(snn_train_acc_array, snn_train_acc)
    snn_test_acc_array = np.append(snn_test_acc_array, snn_test_acc)
    snn_layer1_spikes_count_per_input_array = np.append(snn_layer1_spikes_count_per_input_array, (snn_layer1_spikes_count / num_of_train_data))
    snn_layer2_spikes_count_per_input_array = np.append(snn_layer2_spikes_count_per_input_array, (snn_layer2_spikes_count / num_of_train_data))
    if snn_test_acc > snn_max_test_acc:
        snn_max_test_acc = snn_test_acc
        snn_max_test_acc_epoch = epoch + 1
        snn_best_model = copy.deepcopy(snn)

    print(f'Epoch {epoch+1}')
    print(f'Train phase -> ANN: {ann_train_acc:.2f}     SNN: {snn_train_acc:.2f}')
    print(f'Test phase  -> ANN: {ann_test_acc:.2f}     SNN: {snn_test_acc:.2f} ')
    print(f'Loss: {np.mean(epoch_loss):.5f}')

print(f'Learning is Done! Best test accuracy was {snn_max_test_acc:.2f} in epoch {snn_max_test_acc_epoch}')

torch.save(snn_best_model, f'SNN - {snn_max_test_acc:.2f}.pt')

plt.plot(epoch_loss_array, label='Train loss')
plt.title('SNN Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('SNN loss.png', dpi=200)
plt.clf()
plt.close()

plt.plot(snn_train_acc_array, label='Train')
plt.plot(snn_test_acc_array, label='Test')
plt.title('SNN Accuracies')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('SNN Accuracies.png', dpi=200)
plt.clf()
plt.close()

plt.plot(snn_layer1_spikes_count_per_input_array, label='Spikes count')
plt.title('SNN layer 1 spikes per sample')
plt.xlabel('Epoch')
plt.ylabel('Spikes')
plt.legend()
plt.savefig('SNN layer 1 spikes per sample.png', dpi=200)
plt.clf()
plt.close()

plt.plot(snn_layer2_spikes_count_per_input_array, label='Spikes count')
plt.title('SNN layer 2 spikes per sample')
plt.xlabel('Epoch')
plt.ylabel('Spikes')
plt.legend()
plt.savefig('SNN layer 2 spikes per sample.png', dpi=200)
plt.clf()
plt.close()

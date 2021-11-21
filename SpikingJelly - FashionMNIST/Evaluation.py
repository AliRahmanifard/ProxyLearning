from abc import ABC
import torch
import torch.nn as nn
import numpy as np
import time
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
from sklearn.metrics import confusion_matrix
import itertools
from spikingjelly.clock_driven import surrogate, functional
from spikingjelly.clock_driven.neuron import BaseNode
from matplotlib import pyplot as plt
from tqdm import tqdm


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

test_dataset = FashionMNIST(root=dataset_dir, train=False, transform=ToTensor(), download=False)

test_data_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

num_of_classes = len(test_dataset.classes)
num_of_test_data = len(test_dataset.data)

ann = torch.load('ANN - 89.61.pt')
snn = torch.load('SNN - 84.94.pt')
ann.to(device)
snn.to(device)

simulation_time = snn.simulationTime

if_layer1 = snn.fullyConnected[2]
if_layer2 = snn.fullyConnected[4]

if_layer1.set_monitor(monitor_state=True)
if_layer2.set_monitor(monitor_state=True)

ann.eval()
snn.eval()
ann_start_time = 0
ann_end_time = 0
snn_start_time = 0
snn_end_time = 0
inference_spikes_time = np.zeros(shape=simulation_time, dtype=int)
with torch.no_grad():
    ann_tested_num = 0
    ann_tested_correct = 0
    snn_tested_num = 0
    snn_tested_correct = 0
    ann_all_output = torch.tensor([])
    snn_all_output = torch.tensor([])
    for img, label in tqdm(test_data_loader):
        functional.reset_net(snn)
        img = img.to(device)
        label = label.to(device)

        ann_start_time = time.time()
        outputs = ann(img)
        ann_end_time = time.time()
        predicted_ann = torch.max(outputs.data, 1)[1]
        ann_all_output = torch.cat((ann_all_output, predicted_ann), dim=0)
        ann_tested_correct += (predicted_ann == label).sum().item()
        ann_tested_num += label.numel()

        snn_start_time = time.time()
        output_spikes = snn(img)
        snn_end_time = time.time()
        sorted_output_spikes = torch.sort(output_spikes, dim=-1, descending=True)
        predicted_snn = sorted_output_spikes[1][:, 0]
        snn_all_output = torch.cat((snn_all_output, predicted_snn), dim=0)
        snn_tested_correct += (predicted_snn == label).int().sum().item()
        snn_tested_num += label.numel()

        if_layer2_monitor_s = np.asarray(if_layer2.monitor['s'], dtype=int).T
        for b in range(label.numel()):
            c_break = False
            for t in range(simulation_time):
                for c in range(num_of_classes):
                    if if_layer2_monitor_s[c, b, t] == 1:
                        inference_spikes_time[t] += 1
                        c_break = True
                        break
                if c_break:
                    break

ann_test_acc = 100 * ann_tested_correct / ann_tested_num
snn_test_acc = 100 * snn_tested_correct / snn_tested_num
print(f'ANN test accuracy: {ann_test_acc:.2f}')
print(f'SNN test accuracy: {snn_test_acc:.2f}')
print(f'ANN one batch inference time: {ann_end_time - ann_start_time}')
print(f'SNN one batch inference time: {snn_end_time - snn_start_time}')

time_label = np.array([])
for t in range(simulation_time):
    time_label = np.append(time_label, str(t+1))
plt.bar(time_label, inference_spikes_time)
plt.xlabel('Time step')
plt.ylabel('Samples count')
plt.savefig('SNN spikes in each time step.png', dpi=200)
plt.clf()
plt.close()

ann_all_output = ann_all_output.int()
ann_confusion_matrix = confusion_matrix(test_dataset.targets, ann_all_output)
plt.imshow(ann_confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('ANN Confusion matrix')
plt.colorbar()
tick_marks = np.arange(len(test_dataset.classes))
plt.xticks(tick_marks, test_dataset.classes, rotation=45)
plt.yticks(tick_marks, test_dataset.classes)
fmt = 'd'
thresh = ann_confusion_matrix.max() / 2
for i, j in itertools.product(range(ann_confusion_matrix.shape[0]), range(ann_confusion_matrix.shape[1])):
    plt.text(j, i, format(ann_confusion_matrix[i, j], fmt), horizontalalignment="center", color="white" if ann_confusion_matrix[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig('ANN Confusion matrix.png', dpi=200)
plt.clf()
plt.close()

snn_all_output = snn_all_output.int()
snn_confusion_matrix = confusion_matrix(test_dataset.targets, snn_all_output)
plt.imshow(snn_confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('SNN Confusion matrix')
plt.colorbar()
tick_marks = np.arange(len(test_dataset.classes))
plt.xticks(tick_marks, test_dataset.classes, rotation=45)
plt.yticks(tick_marks, test_dataset.classes)
fmt = 'd'
thresh = snn_confusion_matrix.max() / 2
for i, j in itertools.product(range(snn_confusion_matrix.shape[0]), range(snn_confusion_matrix.shape[1])):
    plt.text(j, i, format(snn_confusion_matrix[i, j], fmt), horizontalalignment="center", color="white" if snn_confusion_matrix[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig('SNN Confusion matrix.png', dpi=200)
plt.clf()
plt.close()

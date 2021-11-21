from abc import ABC
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
from tqdm import tqdm
from spikingjelly.clock_driven import surrogate, functional
from spikingjelly.clock_driven.neuron import BaseNode


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


class ArtificialConvolutionalNet(nn.Module, ABC):
    def __init__(self):
        super().__init__()

        self.staticConvolutional = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(128)
        )

        self.convolutional = nn.Sequential(
            nn.Sigmoid(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(128),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2),  # output: 14 * 14
            # nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(128),
            # nn.ReLU(),
            # nn.MaxPool2d(2, 2)  # output: 7 * 7
        )

        self.fullyConnected = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 14 * 14, 128 * 4 * 4, bias=False),
            nn.Sigmoid(),
            # nn.Linear(128 * 4 * 5, 128 * 3 * 3, bias=False),
            # nn.ReLU(),
            nn.Linear(128 * 4 * 4, 128 * 2 * 1, bias=False),
            nn.Sigmoid(),
            nn.Linear(128 * 2 * 1, 10, bias=False)
        )

    def forward(self, x):
        output = self.staticConvolutional(x)
        output = self.convolutional(output)
        output = self.fullyConnected(output)
        return output


class SpikingConvolutionalNet(nn.Module, ABC):
    def __init__(self, simulationTime, neuronThreshold1, neuronThreshold2, neuronThreshold3, neuronThreshold4, neuronThreshold5):
        super().__init__()
        self.simulationTime = simulationTime
        self.neuronThreshold1 = neuronThreshold1
        self.neuronThreshold2 = neuronThreshold2
        self.neuronThreshold3 = neuronThreshold3
        self.neuronThreshold4 = neuronThreshold4
        self.neuronThreshold5 = neuronThreshold5

        self.staticConvolutional = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(128)
        )

        self.convolutional = nn.Sequential(
            OneSpikeIFNode(v_threshold=self.neuronThreshold1, monitor_state=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(128),
            OneSpikeIFNode(v_threshold=self.neuronThreshold2, monitor_state=True),
            nn.MaxPool2d(2, 2),  # output: 14 * 14
            # nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(128),
            # OneSpikeIFNode(monitor_state=False),
            # nn.MaxPool2d(2, 2)  # output: 7 * 7
        )

        self.fullyConnected = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 14 * 14, 128 * 4 * 4, bias=False),
            OneSpikeIFNode(v_threshold=self.neuronThreshold3, monitor_state=True),
            # nn.Linear(128 * 4 * 5, 128 * 3 * 3, bias=False),
            # OneSpikeIFNode(monitor_state=False),
            nn.Linear(128 * 4 * 4, 128 * 2 * 1, bias=False),
            OneSpikeIFNode(v_threshold=self.neuronThreshold4, monitor_state=True),
            nn.Linear(128 * 2 * 1, 10, bias=False),
            OneSpikeIFNode(v_threshold=self.neuronThreshold5, monitor_state=True)
        )

    def forward(self, x):
        x = self.staticConvolutional(x)
        output_spikes = 0
        for t in range(self.simulationTime):
            output_spikes += (self.fullyConnected(self.convolutional(x))) * (t + 1)

        output_spikes[output_spikes == 0] = self.simulationTime
        output_spikes = (self.simulationTime - output_spikes) / self.simulationTime
        return output_spikes


def init_uniform(m):
    if type(m) == nn.Linear:
        m.weight.data.uniform_(0.0, 0.2)


torch.manual_seed(1)
np.random.seed(1)

device = "cpu"  # "cpu" or "cuda:0"
dataset_dir = "./"  # root directory for saving dataset
batch_size = 1  # batch size
learning_rate = 1e-4  # learning rate
threshold1 = 0.2
threshold2 = 0.2
threshold3 = 0.2
threshold4 = 0.2
threshold5 = 0.2
simulation_time = 16
train_epoch = 30  # training epochs

train_dataset = FashionMNIST(root=dataset_dir, train=True, transform=ToTensor(), download=False)
test_dataset = FashionMNIST(root=dataset_dir, train=False, transform=ToTensor(), download=False)

train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
test_data_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

num_of_classes = len(train_dataset.classes)

ann = ArtificialConvolutionalNet()
snn = SpikingConvolutionalNet(simulationTime=simulation_time, neuronThreshold1=threshold1, neuronThreshold2=threshold2, neuronThreshold3=threshold3, neuronThreshold4=threshold4, neuronThreshold5=threshold5)
# ann.apply(init_uniform)
ann.to(device)
snn.to(device)

if_layer1 = snn.convolutional[0]
if_layer2 = snn.convolutional[2]
if_layer3 = snn.fullyConnected[2]
if_layer4 = snn.fullyConnected[4]
if_layer5 = snn.fullyConnected[6]

params_ann = ann.named_parameters()
params_snn = snn.named_parameters()
dict_params_snn = dict(params_snn)
dict_params_ann = dict(params_ann)
names_snn = list(dict_params_snn.keys())
names_ann = list(dict_params_ann.keys())
for i in range(names_snn.__len__()):
    dict_params_snn[names_snn[i]].data = dict_params_ann[names_ann[i]].data

optimizer_ann = torch.optim.Adam(ann.parameters(), lr=learning_rate, betas=(0.8, 0.99), eps=1e-08, weight_decay=1e-06)

for epoch in range(train_epoch):
    ann.train()
    ann_final_num = 0
    ann_final_correct = 0
    snn_final_num = 0
    snn_final_correct = 0
    snn_top2accuracy = 0
    epoch_loss = 0
    batch_id = -1
    if_layer6_spikes = 0
    temp_spike = np.zeros(shape=simulation_time)

    for img, label in tqdm(train_data_loader, position=0):
        batch_id += 1
        if batch_id % 6 != 0:
            continue
        functional.reset_net(snn)
        img = img.to(device)
        label = label.to(device)
        label_one_hot = nn.functional.one_hot(label, num_of_classes).float()
        label_one_hot = label_one_hot.to(device)

        optimizer_ann.zero_grad()

        outputs = ann(img)
        predicted_ann = torch.max(outputs.data, 1)[1]
        ann_final_num += label.numel()
        ann_final_correct += (predicted_ann == label).sum().item()

        output_spikes = snn(img)
        snn_final_num += label.numel()
        sorted_output_spikes = torch.sort(output_spikes, dim=-1, descending=True)
        first_predict = sorted_output_spikes[1][:, 0]
        second_predict = sorted_output_spikes[1][:, 1]
        snn_final_correct += (first_predict == label).int().sum().item()
        snn_top2accuracy += (first_predict == label.to(device)).int().sum().item()
        snn_top2accuracy += (second_predict == label.to(device)).int().sum().item()
        if_layer1_monitor_s = np.asarray(if_layer1.monitor['s']).T[:, :, :, 0, :]
        if_layer1_spike_count = np.sum(if_layer1_monitor_s)
        if_layer1_monitor_h = np.asarray(if_layer1.monitor['h']).T[:, :, :, 0, :]
        # if_layer1_monitor_v = np.asarray(if_layer1.monitor['v']).T[:, :, :, 0, :]
        if_layer2_monitor_s = np.asarray(if_layer2.monitor['s']).T[:, :, :, 0, :]
        if_layer2_spike_count = np.sum(if_layer2_monitor_s)
        if_layer2_monitor_h = np.asarray(if_layer2.monitor['h']).T[:, :, :, 0, :]
        # if_layer2_monitor_v = np.asarray(if_layer2.monitor['v']).T[:, :, :, 0, :]
        if_layer3_monitor_s = np.asarray(if_layer3.monitor['s']).T[:, 0, :]
        if_layer3_spike_count = np.sum(if_layer3_monitor_s)
        if_layer3_monitor_h = np.asarray(if_layer3.monitor['h']).T[:, 0, :]
        # if_layer3_monitor_v = np.asarray(if_layer3.monitor['v']).T[:, 0, :]
        if_layer4_monitor_s = np.asarray(if_layer4.monitor['s']).T[:, 0, :]
        if_layer4_spike_count = np.sum(if_layer4_monitor_s)
        if_layer4_monitor_h = np.asarray(if_layer4.monitor['h']).T[:, 0, :]
        # if_layer4_monitor_v = np.asarray(if_layer4.monitor['v']).T[:, 0, :]
        if_layer5_monitor_s = np.asarray(if_layer5.monitor['s']).T
        if_layer5_spike_count = np.sum(np.sum(if_layer5_monitor_s, 0), 0)
        temp_spike += if_layer5_spike_count
        if_layer5_monitor_h = np.asarray(if_layer5.monitor['h']).T[:, 0, :]
        # if_layer5_monitor_v = np.asarray(if_layer5.monitor['v']).T[:, 0, :]

        outputs.data.copy_(output_spikes)
        loss = nn.functional.mse_loss(outputs, label_one_hot)
        epoch_loss = loss.data.item()
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
            snn_tested_correct += (output_spikes.max(1)[1] == label).int().sum().item()
            snn_tested_num += label.numel()

    print()
    print(f'Epoch {epoch+1} - Train phase -> ANN: {(100 * ann_final_correct / ann_final_num):.2f}     SNN: {(100 * snn_final_correct / snn_final_num):.2f}     SNN_top2: {(100 * snn_top2accuracy / snn_final_num):.2f}')
    print(f'          Test phase  -> ANN: {(100 * ann_tested_correct / ann_tested_num):.2f}     SNN: {(100 * snn_tested_correct / snn_tested_num):.2f}')
    print(f'          Loss: {epoch_loss}')

print('Learning is Done!')

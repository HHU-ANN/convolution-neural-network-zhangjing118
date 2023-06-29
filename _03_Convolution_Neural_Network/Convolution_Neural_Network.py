# 在该文件NeuralNetwork类中定义你的模型 
# 在自己电脑上训练好模型，保存参数，在这里读取模型参数（不要使用JIT读取），在main中返回读取了模型参数的模型

import os

os.system("sudo pip3 install torch")
os.system("sudo pip3 install torchvision")

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from torch.utils.data import DataLoader
    

class NeuralNetwork(nn.Module):      #Alexnet
    # 在Pytorch中，这块代码是核心，__init__主要包括神经网络的定义，forward方法包括计算的流程
    # 在构造神经网络时，模块需要继承自nn.Module
    def __init__(self, class_num=10, init_weights=False):
        super().__init__()
        # 将全连接层之前的多个层打包
        self.features = nn.Sequential(
            # 为了加快训练，将参数减为原来的一半
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),  # inplace一种增加计算量但是能够降低内存使用的方法
            nn.MaxPool2d(3, 2),
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2)

        )

        # 全连接层部分打包

        self.classifier = nn.Sequential(
            # dropout一般是放在全连接层和全连接层之间
            nn.Dropout(p=0.5),  # p随机失活的比例，默认值为0.5
            nn.Linear(256 * 6 * 6, 2048 * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048 * 2, 2048 * 2),
            nn.ReLU(inplace=True),
            nn.Linear(2048 * 2, class_num),
        )

        # 初始化权重
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.feature(x)
        x = torch.flatten(x, start_dim=1)  # 展平处理，从channel这个维度开始进行展平
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def read_data():
    dataset_train = torchvision.datasets.CIFAR10(root='../data/exp03', train=True, download=True, transform=torchvision.transforms.ToTensor())
    dataset_val = torchvision.datasets.CIFAR10(root='../data/exp03', train=False, download=False, transform=torchvision.transforms.ToTensor())
    data_loader_train = DataLoader(dataset=dataset_train, batch_size=256, shuffle=True)
    data_loader_val = DataLoader(dataset=dataset_val, batch_size=256, shuffle=False)
    return dataset_train, dataset_val, data_loader_train, data_loader_val

def main():
    model = NeuralNetwork(10,init_weights=True) # 若有参数则传入参数
    # criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数，用于分类问题
    # optimizer = optim.SGD(model.parameters(), lr=0.1)  # SGD优化器
    #dataset_train = torchvision.datasets.CIFAR10(root='../data/exp03', train=True, download=True, transform=torchvision.transforms.ToTensor())
    # dataset_val = torchvision.datasets.CIFAR10(root='../data/exp03', train=False, download=False, transform=torchvision.transforms.ToTensor())
    # data_loader_train = DataLoader(dataset=dataset_train, batch_size=256, shuffle=True)
    # data_loader_val = DataLoader(dataset=dataset_val, batch_size=256, shuffle=False)
    #torch.save(model.state_dict(), 'model.pth')
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    model.load_state_dict(torch.load(parent_dir + '/pth/model.pth',map_location='cpu'))
    return model
    
# # 在该文件NeuralNetwork类中定义你的模型
# # 在自己电脑上训练好模型，保存参数，在这里读取模型参数（不要使用JIT读取），在main中返回读取了模型参数的模型
# import os
#
# os.system("sudo pip3 install torch")
# os.system("sudo pip3 install torchvision")
#
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torchvision
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader
#
#
#
# class NeuralNetwork(nn.Module):
#     def __init__(self, class_num=1000, init_weights=False):
#         super().__init__()
#         # 将全连接层之前的多个层打包
#         self.feature = nn.Sequential(
#             nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),  # input[3, 224, 224]  output[48, 55, 55]
#             nn.ReLU(inplace=True),  # inplace一种增加计算量但是能够降低内存使用的方法
#             nn.MaxPool2d(3, 2),  # [48, 27, 27]
#             nn.Conv2d(48, 128, kernel_size=5, stride=1, padding=2),  # [128, 27, 27]
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(3, 2),  # [128, 13, 13]
#             nn.Conv2d(128, 192, kernel_size=3, stride=1, padding=1),  # [192, 13, 13]
#             nn.ReLU(inplace=True),
#             nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),  # [192, 13, 13]
#             nn.ReLU(inplace=True),
#             nn.Conv2d(192, 128, kernel_size=3, stride=1, padding=1),  # [128, 13, 13]
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(3, 2)  # [128, 6, 6]
#         )
#         # 全连接层部分打包
#         self.classifier = nn.Sequential(
#             # dropout一般是放在全连接层和全连接层之间
#             nn.Dropout(p=0.5),  # p随机失活的比例，默认值为0.5
#             nn.Linear(128 * 6 * 6, 2048),
#             nn.ReLU(inplace=True),
#             nn.Dropout(p=0.5),
#             nn.Linear(2048, 2048),
#             nn.ReLU(inplace=True),
#             nn.Linear(2048, class_num),
#         )
#         # 初始化权重
#         if init_weights:
#             self._initialize_weights()
#
#     def forward(self, x):
#         x = self.feature(x)
#         x = torch.flatten(x, start_dim=1)  # 展平处理，从channel这个维度开始进行展平
#         x = self.classifier(x)
#         return x
#
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, 0, 0.01)
#
# def read_data():
#     # 这里可自行修改数据预处理，batch大小也可自行调整
#     # 保持本地训练的数据读取和这里一致
#     dataset_train = torchvision.datasets.CIFAR10(root='../data/exp03', train=True, download=True, transform=torchvision.transforms.ToTensor())
#     dataset_val = torchvision.datasets.CIFAR10(root='../data/exp03', train=False, download=False, transform=torchvision.transforms.ToTensor())
#     data_loader_train = DataLoader(dataset=dataset_train, batch_size=256, shuffle=True)
#     data_loader_val = DataLoader(dataset=dataset_val, batch_size=256, shuffle=False)
#     return dataset_train, dataset_val, data_loader_train, data_loader_val
#
# def main():
#     model = NeuralNetwork(class_num=10,init_weights=True) # 若有参数则传入参数
#     torch.save(model.state_dict(), 'model.pth')
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     parent_dir = os.path.dirname(current_dir)
#     model.load_state_dict(torch.load(parent_dir + '/pth/model.pth',map_location='cpu'))
#     return model
# 在该文件NeuralNetwork类中定义你的模型
# 在自己电脑上训练好模型，保存参数，在这里读取模型参数（不要使用JIT读取），在main中返回读取了模型参数的模型


import os

os.system("sudo pip3 install torch")
os.system("sudo pip3 install torchvision")





import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
num_epochs = 25  # 50轮
batch_size = 50  # 50步长
learning_rate = 0.015  # 学习率0.01
from torch.utils.data import DataLoader
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

def read_data():
    # 这里可自行修改数据预处理，batch大小也可自行调整
    # 保持本地训练的数据读取和这里一致
    transform = transforms.Compose([
        transforms.Pad(4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),
        transforms.ToTensor()])
    dataset_train = torchvision.datasets.CIFAR10(root='../data/exp03', train=True, download=True, transform=torchvision.transforms.ToTensor())
    dataset_val = torchvision.datasets.CIFAR10(root='../data/exp03', train=False, download=False, transform=torchvision.transforms.ToTensor())
    data_loader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)
    data_loader_val = DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=False)
    return dataset_train, dataset_val, data_loader_train, data_loader_val

# 3x3 卷积定义
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)

   # Resnet 的残差块
class ResidualBlock(nn.Module):
        def __init__(self, in_channels, out_channels, stride=1, downsample=None):
            super(ResidualBlock, self).__init__()
            self.conv1 = conv3x3(in_channels, out_channels, stride)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = conv3x3(out_channels, out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.downsample = downsample

        def forward(self, x):
            residual = x
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.bn2(out)
            if self.downsample:
                residual = self.downsample(x)
            out += residual
            out = self.relu(out)
            return out

class NeuralNetwork(nn.Module):
    # ResNet定义
    def __init__(self, block, layers, num_classes=10):
        super(NeuralNetwork, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[1], 2)
        self.layer3 = self.make_layer(block, 64, layers[2], 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out





def main():
    model1 = NeuralNetwork(ResidualBlock, [2, 2, 2]).to(device) # 若有参数则传入参数
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    model1.load_state_dict(torch.load(parent_dir + '/pth/model.pth'))
    return model1
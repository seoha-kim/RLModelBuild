import torch
from torch.nn import Conv2d, BatchNorm2d, Linear
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as F

class DQNbn(nn.Module):
    def __init__(self, in_channels=4, n_actions=14):
        """
        Initialize Deep Q Network with Batch Normalization

        :param in_channels: number of input channels
        :param n_actions: number of outputs
        """
        super(DQNbn, self).__init__()
        self.conv1 = Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.bn1 = BatchNorm2d(32)
        self.conv2 = Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = BatchNorm2d(64)
        self.conv3 = Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = BatchNorm2d(64)
        self.fc4 = Linear(7*7*64, 512)
        self.head = Linear(512, n_actions)

    def forward(self, x):
        x = x.float() / 255
        x = F.relu(self.bn1(self.conv1))
        x = F.relu(self.bn2(self.conv2))
        x = F.relu(self.bn3(self.conv3))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        return self.head(x)

class DQN(nn.Module):
    def __init__(self, in_channels=4, n_actions=14):
        """
        Initialize Deep Q Network

        :param in_channels:
        :param n_actions:
        """
        super(DQN, self).__init__()
        self.conv1 = Conv2d(in_channels, 32, kernel_size=8, stride=4)
        # self.bn1 = BatchNorm2d(32)
        self.conv2 = Conv2d(32, 64, kernel_size=4, stride=2)
        # self.bn2 = BatchNorm2d(64)
        self.conv3 = Conv2d(64, 64, kernel_size=3, stride=1)
        # self.bn3 = BatchNorm2d(64)
        self.fc4 = Linear(7 * 7 * 64, 512)
        self.head = Linear(512, n_actions)

    def forward(self, x):
        x = x.float() / 255
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        return self.head(x)

# https://github.com/jmichaux/dqn-pytorch/
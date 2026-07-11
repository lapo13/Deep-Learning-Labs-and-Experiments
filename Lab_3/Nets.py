import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    

class DCQN(nn.Module):
    def __init__(self, input_shape, action_size):
        super(DCQN, self).__init__()
        in_channels = input_shape[0]

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.maxpool1 = nn.MaxPool2d(kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        conv_out_size = self._get_conv_output(input_shape)

        self.fc1 = nn.Linear(conv_out_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, action_size)

    def _forward_conv(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)
        return x

    def _get_conv_output(self, shape):
        with torch.no_grad():
            o = torch.zeros(1, *shape)
            o = self._forward_conv(o)
            return int(torch.prod(torch.tensor(o.size()[1:]))) #dinamic dimensioning of the output size of conv layers, excluding the batch dimension

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
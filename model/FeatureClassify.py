import random

import numpy as np
import torch
import torch.nn as nn

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


# 2. 定义模型-version2
class ModelWithoutFC(nn.Module):
    def __init__(self):
        super(ModelWithoutFC, self).__init__()
        self.CnnBlock = nn.Sequential(
            # first layer cnn
            nn.Conv1d(in_channels=5, out_channels=32, kernel_size=(3,), stride=(1,), padding=4),
            nn.ReLU(inplace=True),
            # second layer cnn
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=(3,), stride=(1,)),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=0.3,inplace=False),
            # third layer cnn
            nn.Conv1d(in_channels=16, out_channels=8, kernel_size=(3,), stride=(1,)),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=0.3,inplace=False)
            nn.Conv1d(in_channels=8, out_channels=32, kernel_size=(3,), stride=(1,)),
            nn.ReLU(inplace=True),
        )
        self.LstmBlock = nn.Sequential(
            nn.LSTM(input_size=32, hidden_size=16, num_layers=1, bidirectional=True),
        )

        self.DenseBlock2 = nn.Sequential(
            nn.Linear(in_features=32, out_features=5),
            # TODO 此处有问题
        )

    def forward(self, x):
        x = self.CnnBlock(
            # 1D CNN是在最后一个维度上扫
            x.permute(0, 2, 1)
        )
        # 把维度交换回来
        x = x.permute(0, 2, 1)
        x, (h, c) = self.LstmBlock(x)
        x, (h, c) = self.LstmBlock(x)
        x = self.DenseBlock2(x)
        m1 = nn.Softmax(dim=2)
        x = m1(x)
        return x

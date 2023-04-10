import torch
import torch.nn as nn

from utils.common import Conv1d


class VGG(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()

        self.features = nn.Sequential(
            Conv1d(1, 64, k=3, s=1, p=1),
            Conv1d(64, 64, k=3, s=1, p=1),
            nn.MaxPool1d(kernel_size=2, stride=2),

            Conv1d(64, 128, k=3, s=1, p=1),
            Conv1d(128, 128, k=3, s=1, p=1),
            nn.MaxPool1d(kernel_size=2, stride=2),

            # Conv1(128, 256, k=3, s=1, p=1),
            # Conv1(256, 256, k=3, s=1, p=1),
            # Conv1(256, 256, k=3, s=1, p=1),
            # nn.MaxPool1d(kernel_size=2, stride=2),
            #
            # Conv1(256, 512, k=3, s=1, p=1),
            # Conv1(512, 512, k=3, s=1, p=1),
            # Conv1(512, 512, k=3, s=1, p=1),
            # nn.MaxPool1d(kernel_size=2, stride=2),
            #
            # Conv1(512, 512, k=3, s=1, p=1),
            # Conv1(512, 512, k=3, s=1, p=1),
            # Conv1(512, 512, k=3, s=1, p=1),
            # nn.MaxPool1d(kernel_size=2, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool1d(3)

        self.classifier = nn.Sequential(
            nn.Linear(128 * 3, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

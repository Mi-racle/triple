import torch
import torch.nn as nn


import utils.common as cm


class Backbone(nn.Module):
    def __init__(self, num=100):
        super().__init__()
        self.net = nn.Sequential(
            cm.Conv2d(3, 64, 6, 2, 2),
            cm.Conv2d(64, 128, 3, 2),
            cm.C3(128, 128),
            cm.Conv2d(128, 256, 3, 2),
            cm.C3(256, 256),
            cm.C3(256, 256),
            cm.Conv2d(256, 512, 3, 2),
            cm.C3(512, 512),
            cm.C3(512, 512),
            cm.C3(512, 512),
            cm.Conv2d(512, 1024, 3, 2),
            cm.C3(1024, 1024),
            cm.SPPF(1024, 1024, 5),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num),
        )

    def forward(self, x):
        x = x.transpose(0, 1)

        view_pool = []

        for v in x:
            v = self.features(v)
            v = v.view(v.size(0), 256 * 6 * 6)

            view_pool.append(v)

        pooled_view = view_pool[0]
        for i in range(1, len(view_pool)):
            pooled_view = torch.max(pooled_view, view_pool[i])

        pooled_view = self.classifier(pooled_view)
        return pooled_view


import numpy as np
import random
import torch
import torchvision
import torch.nn as nn
from torch.cuda import amp
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


from utils.vgg import VGG


def run():
    model = VGG(num_classes=3)
    curves = generate(100)
    # plt.plot(np.arange(0, 100, 0.1).tolist(), torch.reshape(curves[0][0], (1000, 1)).tolist())
    # plt.show()

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(500):
        pred, truth = [], []
        for curve in curves:
            output = model(curve[0])
            pred.append(torch.argmax(output).item())
            truth.append(curve[1])
        pred = torch.tensor(pred, dtype=torch.float32, requires_grad=True)
        truth = torch.tensor(truth, dtype=torch.float32)
        loss = loss_fn(pred, truth)
        print(str(epoch) + ': ' + str(loss.item()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def main():
    run()


if __name__ == '__main__':
    main()

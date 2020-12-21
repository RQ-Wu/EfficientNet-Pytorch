from torchvision import datasets
from torchvision import transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch
import argparse
import time
from model import *


def train_net(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')  # cuda or cpu

    # load dataset
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    train_dataset = datasets.CIFAR100(root='dataset/', train=True, download=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # load model
    model = EfficientNet(num_classes=args.classes, image_size=args.image_size)
    model = model.to(device)

    # loss function(cross-entropy loss)
    loss_f = torch.nn.CrossEntropyLoss()

    # RMSProp optimizer with decay 0.9 and momentum 0.9
    # we use adam here
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    print("train begin")
    for epoch in range(10000):
        loss_avg = 0
        for iteration, data in enumerate(train_loader):
            # t1 = time.time()
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            # predict
            label_pre = model(images)
            # calculate loss
            loss = loss_f(label_pre, labels)
            loss_avg = (loss_avg * iteration + loss.data) / (iteration + 1)
            #bp
            optim.zero_grad()
            loss.backward()
            optim.step()

            if iteration % 100 == 0:
                print(f'epoch:{epoch}, iter:{iteration}, loss:{loss.data:.3f}, avg_loss:{loss_avg:.3f}')
        # save model every epoch
        torch.save(model.state_dict(), f'logs/Efficient-B0-{epoch}-{loss_avg:.3f}.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=0.04)
    parser.add_argument('--image_size', type=int, default=32)
    parser.add_argument('--classes', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)

    var_args = parser.parse_args()
    train_net(var_args)

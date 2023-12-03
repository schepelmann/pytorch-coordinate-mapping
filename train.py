import argparse
import os

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data.dataloader import DataLoader

from dataset import CoordinateDataset
from network import NeuralNetwork


def train(net: nn.Module, trainloader: DataLoader, output_dir: str) -> None:
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(100):
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            inputs, labels = data

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # print statistics after each epoch
        print(f'Epoch: {epoch + 1},  loss: {running_loss / len(trainloader):.4f}')

    # save the model after training
    torch.save(net.state_dict(), os.path.join(output_dir, './saved_model.pt'))


def parse_opt() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data', '-d',
        type=str,
        default='./data.csv',
        help='Path to the CSV file containing the training data.'
    )
    parser.add_argument(
        '--output_dir', '-o',
        type=str,
        default='./',
        help='Directory to store the trained network in.'
    )

    return parser.parse_args()


def main(opt: argparse.Namespace) -> None:
    train_dataset = CoordinateDataset(opt.data)
    trainloader = DataLoader(dataset=train_dataset, batch_size=16)

    net = NeuralNetwork()

    train(net, trainloader, opt.output_dir)


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
# Copyright (c) 2023 David Boetius
# Licensed under the MIT license
from argparse import ArgumentParser
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import ConcatDataset, TensorDataset, DataLoader

from adult import Adult


if __name__ == "__main__":
    torch.manual_seed(452765764742256)

    parser = ArgumentParser("Train Adult Network")
    parser.add_argument(
        "--name",
        type=str,
        default="adult_net",
        help="Name of the network to train. Used for the network file name.",
    )
    parser.add_argument(
        "--balance_dataset",
        action="store_true",
        help="Whether to make the dataset fair by duplicating the data and flipping "
        "the 'sex' attribute on the copy.",
    )
    parser.add_argument(
        "--force_fair",
        action="store_true",
        help="Whether to make the network dependency fair by making it indifferent "
        "to the 'sex' attribute (zeroing the weights of this attribute).",
    )
    args = parser.parse_args()

    network_path = Path("adult", args.name + ".pyt")
    if network_path.exists():
        print("File where to save network already exists.")
        print(
            "Hit enter to continue training and overwrite existing file, "
            "press CTRL+C to abort."
        )
        input("> ??? ")

    train_set = Adult(".datasets", download=True)
    test_set = Adult(".datasets", train=False, download=True)

    if args.balance_dataset:
        inputs_copy = train_set.data.clone()
        sensitive_attrs = train_set.sensitive_column_indices
        inputs_copy[:, sensitive_attrs] = 1.0 - inputs_copy[:, sensitive_attrs]
        train_copy = TensorDataset(inputs_copy, train_set.targets)
        train_set = ConcatDataset((train_set, train_copy))

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    epoch_len = len(train_loader)
    test_loader = DataLoader(test_set, batch_size=1024, shuffle=False)
    train_loader2 = DataLoader(train_set, batch_size=1024, shuffle=True)
    loss_fn = nn.CrossEntropyLoss()

    @torch.no_grad()
    def accuracy(loader):
        acc = 0.0
        for inputs, targets in loader:
            preds = network(inputs)
            acc += (preds.argmax(dim=-1) == targets).float().mean()
        return acc / len(loader)

    network = nn.Sequential(
        nn.Linear(test_set.data.size(-1), 100),
        nn.ReLU(),
        nn.Linear(100, 10),
        nn.ReLU(),
        nn.Linear(10, 2),
    )
    optim = torch.optim.Adam(network.parameters(), lr=0.01)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optim, milestones=(epoch_len, 2 * epoch_len), gamma=0.25
    )

    num_epochs = 5
    for epoch in range(num_epochs):
        for i, (inputs, targets) in enumerate(iter(train_loader)):
            optim.zero_grad()
            preds = network(inputs)
            loss = loss_fn(preds, targets)
            loss.backward()
            optim.step()
            lr_scheduler.step()

            if i % 100 == 0 or i == len(train_loader) - 1:
                train_acc = accuracy(train_loader2)
                test_acc = accuracy(test_loader)
                print(
                    f"[Epoch {epoch+1}/{num_epochs} ({i / epoch_len * 100:4.1f}%)] "
                    f"train loss: {loss:3.4f}, "
                    f"train accuracy: {train_acc*100:3.1f}%, "
                    f"test accuracy: {test_acc*100:3.1f}%."
                )

    print("Training finished.")

    if args.force_fair:
        print("Making network agnostic to sensitive attributes.")
        # set weights of sensitive attributes of all neurons of the first layer
        network[0].weight.data[:, test_set.sensitive_column_indices] = 0.0

    train_acc = accuracy(train_loader2)
    test_acc = accuracy(test_loader)
    print(
        f"Final results\n"
        f"=====================\n"
        f"Train accuracy: {train_acc*100:3.1f}%\n"
        f"Test accuracy:  {test_acc*100:3.1f}%\n"
        f"====================="
    )

    network_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(network, network_path)

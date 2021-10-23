# Copyright 2021 Mark Chilenski
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from contextlib import nullcontext

import torch


def run_training_loop(
    train_loader, val_loader, model, loss_fn, device, optimizer, num_epoch
):
    """Run the training process for multiple epochs.

    Parameters
    ----------
    train_loader : DataLoader
        DataLoader containing the training data to use.
    val_loader : DataLoader
        DataLoader containing the validation data to use.
    model : nn.Module
        Network to train.
    loss_fn : nn.Module
        Loss function to optimize.
    device : torch.device
        Device to train on.
    optimizer : optim.Optimizer
        Optimizer to use to train the network.
    num_epoch : int
        The number of epochs to train for.
    """
    for i_epoch in range(num_epoch):
        run_single_epoch(
            train_loader, val_loader, model, loss_fn, device, optimizer,
            i_epoch
        )


def run_single_epoch(
    train_loader, val_loader, model, loss_fn, device, optimizer, i_epoch
):
    """Run a single  epoch, consisting of a training epoch and a validation
    epoch.

    Parameters
    ----------
    train_loader : DataLoader
        DataLoader containing the training data to use.
    val_loader : DataLoader
        DataLoader containing the validation data to use.
    model : nn.Module
        Network to train.
    loss_fn : nn.Module
        Loss function to optimize.
    device : torch.device
        Device to train on.
    optimizer : optim.Optimizer
        Optimizer to use to train the network.
    i_epoch : int
        The epoch number.
    """
    train_loss, train_acc = epoch(
        train_loader, model, loss_fn, device, optimizer=optimizer
    )
    val_loss, val_acc = epoch(val_loader, model, loss_fn, device)

    print(f'\nEpoch {i_epoch}:\tLoss\tAcc.')
    print(
        f'Train:\t\t{train_loss:.3f}\t{train_acc:.3f}'
    )
    print(
        f'Val:\t\t{val_loss:.3f}\t{val_acc:.3f}'
    )


def epoch(loader, model, loss_fn, device, optimizer=None):
    """Run all samples in the given loader through the network, optionally
    doing training.

    Parameters
    ----------
    loader : DataLoader
        DataLoader containing the data to use.
    model : nn.Module
        Network to train.
    loss_fn : nn.Module
        Loss function to optimize.
    device : torch.device
        Device to train on.
    optimizer : optim.Optimizer, optional
        Optimizer to use to train the network. If no optimizer is present,
        training will not be run.

    Returns
    -------
    loss : float
        Mean loss per sample across the epoch.
    accuracy : float
        Accuracy across the epoch.
    """
    if optimizer:
        model.train()
        cm = nullcontext()
    else:
        model.eval()
        cm = torch.no_grad()

    with cm:
        sample_count = 0
        accumulated_loss = 0.0
        accumulated_num_correct = 0.0
        for x, labels in loader:
            x = x.to(device)
            labels = labels.to(device)

            logits = model(x)
            loss = loss_fn(logits, labels)

            labels_pred = torch.argmax(logits.detach(), dim=-1)
            num_correct = (labels_pred == labels).sum()

            if optimizer:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            sample_count += x.shape[0]
            accumulated_loss += loss.detach().cpu().item()
            accumulated_num_correct += num_correct.detach().cpu().item()

    return (
        accumulated_loss / sample_count,
        accumulated_num_correct / sample_count
    )

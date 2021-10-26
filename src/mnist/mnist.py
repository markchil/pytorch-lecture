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

import torch
import torch.nn as nn
import torch.optim as optim

from data import get_data
from model import MLPNet
from training import run_training_loop, epoch


def get_device():
    """Get the device to run training/inference on. Defaults to the GPU, if one
    is available.

    Returns
    -------
    device : torch.device
        The device to use.
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    return device


if __name__ == '__main__':
    torch.manual_seed(20211026)

    device = get_device()

    train_loader, val_loader, test_loader = get_data(
        drop_last=True, batch_size=128
    )

    num_class = 10
    layer_widths = [256, 128, 64]
    model = MLPNet(train_loader.dataset[0][0].shape, num_class, layer_widths)
    model.to(device)

    optimizer = optim.Adam(model.parameters())

    loss_fn = nn.CrossEntropyLoss(reduction='sum')

    num_epoch = 10
    run_training_loop(
        train_loader, val_loader, model, loss_fn, device, optimizer, num_epoch
    )
    torch.save(model.state_dict(), 'model.pt')

    test_loss, test_acc = epoch(test_loader, model, loss_fn, device)
    print('\n\t\tLoss\tAcc.')
    print(f'Test:\t\t{test_loss:.3f}\t{test_acc:.3f}')

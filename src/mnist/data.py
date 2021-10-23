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

from torch.utils.data import random_split, DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms


def get_data(
    root='./data', val_fraction=0.05, train_shuffle=True, **dataloader_kwargs
):
    """Get the train, val, and test data.

    Parameters
    ----------
    root : str, optional
        The path to download/load the data from. Default is './data'.
    val_fraction : float, optional
        The fraction of training data to reserve as a validation set. Default
        is 0.05.
    train_shuffle : bool, optional
        Whether or not to shuffle the training set. Default is True.
    **dataloader_kwargs
        All additional keyword arguments are passed to the constructor of
        DataLoader. The same keyword arguments are used for all three sets.

    Returns
    -------
    train_dataloader, val_dataloader, test_dataloader : DataLoader
        DataLoaders containing the training, validation, and test data.
    """
    train_dataset, val_dataset, test_dataset = get_datasets(root, val_fraction)
    train_dataloader = DataLoader(
        train_dataset, shuffle=train_shuffle, **dataloader_kwargs
    )
    val_dataloader = DataLoader(val_dataset, **dataloader_kwargs)
    test_dataloader = DataLoader(test_dataset, **dataloader_kwargs)

    return train_dataloader, val_dataloader, test_dataloader


def get_datasets(root, val_fraction):
    """Load the MNIST datasets.

    Parameters
    ----------
    root : str
        The path to download/load the data from.
    val_fraction : float
        The fraction of training data to reserve as a validation set.

    Returns
    -------
    train_dataset, val_dataset, test_dataset : Dataset
        Datasets containing the training, validation, and test data.
    """
    train_val_dataset = MNIST(
        root, train=True, download=True, transform=transforms.ToTensor()
    )
    train_dataset, val_dataset = train_val_split(
        train_val_dataset, val_fraction
    )

    test_dataset = MNIST(
        root, train=False, download=True, transform=transforms.ToTensor()
    )

    return train_dataset, val_dataset, test_dataset


def train_val_split(dataset, val_fraction):
    """Split a given dataset into a training and validation set.

    Parameters
    ----------
    dataset : Dataset
        The dataset to split.
    val_fraction : float
        The fraction of training data to reserve as a validation set.

    Returns
    -------
    train_dataset, val_dataset : Dataset
        Datasets containing the training and validation data.
    """
    val_samples = int(len(dataset) * val_fraction)
    train_samples = len(dataset) - val_samples

    train_dataset, val_dataset = random_split(
        dataset, [train_samples, val_samples]
    )

    return train_dataset, val_dataset

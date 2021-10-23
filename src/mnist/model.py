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


def make_mlp_stack(in_features, layer_widths):
    """Make a stack of alternating Linear and ReLU layers (i.e., a multilayer
    perceptron (MLP)).

    Parameters
    ----------
    in_features : int
        The number of input features to the stack.
    layer_widths : list of int
        The number of output features from each Linear layer in the stack.

    Returns
    -------
    stack : nn.Module
        The stack of layers.
    """
    layers = []
    for out_features in layer_widths:
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        in_features = out_features

    return nn.Sequential(*layers)


class MLPNet(nn.Module):
    def __init__(
        self, input_shape, num_class, layer_widths
    ):
        """Neural network consisting of a flatten operation, MLP stack, and
        a linear output layer.

        Parameters
        ----------
        input_shape : torch.Size
            The shape of a single (unbatched) input. The n-dimensional input
            will be flattened.
        num_class : int
            The number of classes. (This sets the number of output units in the
            final layer).
        layer_widths : list of int
            The number of output features from each Linear layer in the MLP
            stack. Note that the linear output layer is separate from the MLP
            stack, so there is *no* need for the last entry to be equal to
            num_class.
        """
        super().__init__()

        flatten = nn.Flatten()
        mlp_stack = make_mlp_stack(
            self.get_mlp_in_features(input_shape), layer_widths
        )
        output_layer = nn.Linear(layer_widths[-1], num_class)
        # NOTE: we do *not* apply softmax here, as it is more stable to allow
        # the loss function to take care of that.
        self.net = nn.Sequential(flatten, mlp_stack, output_layer)

    def get_mlp_in_features(self, input_shape):
        """Get the number of features which will be passed to the MLP stack for
        a given input shape.

        Parameters
        ----------
        input_shape : torch.Size
            The shape of a single (unbatched) input. The n-dimensional input
            will be flattened.

        Returns
        -------
        in_features : int
            The number of input features to the MLP stack.
        """
        return torch.prod(torch.as_tensor(input_shape))

    def forward(self, x):
        """Evaluate the network on a given (batch) of inputs.

        Parameters
        ----------
        x : torch.Tensor of torch.float, (num_samp, num_channel, height, width)
            The input images.

        Returns
        -------
        logits : torch.Tensor of torch.float, (num_samp, num_class)
            The output logits.
        """
        return self.net(x)

import torch
import torch.nn as nn


class ScaledLinear(nn.Module):
    def __init__(self, in_features, out_features, scale):
        super().__init__()

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.register_buffer('scale', torch.as_tensor(scale))

    def forward(self, x):
        return self.scale * torch.nn.functional.linear(
            x, self.weight, self.bias
        )


layer = ScaledLinear(16, 4, 2.0)
x = torch.rand(1, 16)
out = layer(x)
print(out.shape)  # torch.Size([1, 4])
print(layer.state_dict().keys())
# odict_keys(['weight', 'bias', 'scale'])

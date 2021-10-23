import torch
from torch.utils.data import TensorDataset

x = torch.rand(500, 10, dtype=torch.float)
y = torch.randint(low=0, high=5, size=(500,), dtype=torch.long)
dataset = TensorDataset(x, y)

x_0, y_0 = dataset[0]
print(x_0.shape)  # torch.Size([10])
print(y_0)  # tensor(1)

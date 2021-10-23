import torch
from torch.utils.data import TensorDataset, DataLoader

x = torch.rand(500, 10, dtype=torch.float)
y = torch.randint(low=0, high=5, size=(500,), dtype=torch.long)
dataset = TensorDataset(x, y)
loader = DataLoader(dataset, batch_size=5, shuffle=True)

x_0, y_0 = next(iter(loader))
print(x_0.shape)  # torch.Size([5, 10])
print(y_0.shape)  # torch.Size([5])

import torch
x = torch.rand(3, 4, 5)  # Make a random Tensor
print(x.shape)  # torch.Size([3, 4, 5])
print(x[1, 0, 2])  # tensor(0.3494)
print(x[:, 0, 0])
# tensor([0.9643, 0.2863, 0.1553])

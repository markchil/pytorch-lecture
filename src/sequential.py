import torch.nn as nn

net = nn.Sequential(
    nn.Linear(16, 128),  # 16 in, 128 out
    nn.ReLU(),  # "ReLU" activation function
    nn.Linear(128, 4)  # 128 in, 4 out
)

x = torch.rand(1, 16)
out = net(x)
print(out.shape)  # torch.Size([1, 4])

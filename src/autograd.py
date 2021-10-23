import torch

x = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(1.0, requires_grad=True)
z = x ** 2 + x + y
z.backward()  # Compute the gradients
print(x.grad)  # tensor(3.)
print(y.grad)  # tensor(1.)

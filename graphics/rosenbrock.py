import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim

mpl.rc('font', size=18)
plt.ion()
plt.close('all')

x0 = -0.5
y0 = 0.5

x = torch.tensor(x0, requires_grad=True, dtype=torch.double)
y = torch.tensor(y0, requires_grad=True, dtype=torch.double)

optimizer = optim.SGD([x, y], lr=0.005)

a = 1.0
b = 100.0

steps = 100
x_history = torch.empty(steps, dtype=x.dtype)
y_history = torch.empty(steps, dtype=y.dtype)

for i_step in range(steps):
    x_history[i_step] = x.detach()
    y_history[i_step] = y.detach()

    f = (a - x) ** 2 + b * (y - x ** 2) ** 2

    optimizer.zero_grad()
    f.backward()
    optimizer.step()

f, ax = plt.subplots()
ax.plot(x_history.numpy(), y_history.numpy(), '.:')
ax.plot(x0, y0, 'o')

# x_low, x_hi = ax.get_xlim()
# y_low, y_hi = ax.get_ylim()
x_low = -1
x_hi = 1
y_low = -1
y_hi = 1
x_grid = np.linspace(x_low, x_hi, 1000)
y_grid = np.linspace(y_low, y_hi, 1001)
X, Y = np.meshgrid(x_grid, y_grid)
F = (a - X) ** 2 + b * (Y - X ** 2) ** 2

ax.contour(X, Y, np.log10(F), colors='gray', linestyles='-', levels=10)
ax.set_aspect('equal')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_title('Gradient Descent on\nRosenbrock Function')

f.savefig('rosenbrock.png', dpi=300, bbox_inches='tight')

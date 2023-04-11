import torch

x = torch.linspace(-1, 1, 5)[:-1]
y = torch.linspace(-1, 1, 5)[:-1]
xx, yy = torch.meshgrid(x, y)
xx = xx.reshape(-1, 1)
yy = yy.reshape(-1, 1)
print(torch.cat([xx, yy], dim=1))

# print(a)
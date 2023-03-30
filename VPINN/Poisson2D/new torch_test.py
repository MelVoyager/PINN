import torch
from net_class import MLP
from tqdm import tqdm
import os, sys
import matplotlib.pyplot as plt
import new_lengendre

os.chdir(sys.path[0])

def u(x, y):
    return (0.1 * torch.sin(2 * torch.pi * x) + torch.tanh(10 * x)) * torch.sin(2 * torch.pi * y)

def f(x, y, m=1, n=1, c=0.1, k=10):
    term1 = (4 * torch.pi**2 * m**2 * c * torch.sin(2 * torch.pi * m * x) +
             (2 * k**2 * torch.sinh(k * x) / torch.cosh(k * x)**3)) * torch.sin(2 * torch.pi * n * y)
    term2 = 4 * torch.pi**2 * n**2 * (c * torch.sin(2 * torch.pi * m * x) + torch.tanh(k * x)) * torch.sin(2 * torch.pi * n * y)
    return -(term1 + term2)

def gradient_u(x, y):
    dx = (0.2 * torch.pi * torch.cos(2*torch.pi*x) + 10 * (1 / torch.cosh(10*x))**2) * torch.sin(2 * torch.pi * y)
    dy = 2 * torch.pi * (0.1 * torch.sin(2*torch.pi*x) + torch.tanh(10*x)) * torch.cos(2*torch.pi*y)
    return torch.cat([dx, dy], dim=1)

N = 100
eps = torch.rand(1).item() * 1e-3
x = torch.linspace(-1+eps, 1-eps, N)
y = torch.linspace(-1+eps, 1-eps, N)
xx, yy = torch.meshgrid(x, y, indexing='ij')
xx = (xx.reshape(-1, 1)).requires_grad_(True)
yy = (yy.reshape(-1, 1)).requires_grad_(True)

s1 = torch.sum(gradient_u(xx, yy) * new_lengendre.v(1, 2, xx, yy))
s2 = torch.sum(f(xx, yy) * new_lengendre.v(0, 2, xx, yy))

print(s1, s2)
# plt.imshow((new_lengendre.v(1, 3, xx, yy)[:,0]).reshape(N, N).detach().numpy())
# plt.colorbar()
# plt.show()
# s1 = torch.sum(gradient_u(xx, yy) * new_lengendre.v(1, 2))
# s2 = torch.sum(f(xx, yy) * v(xx, yy, 2))
# s3 = torch.sum(du * gradient_v(xx, yy, 2))
# print(s1, s2, s3)
# import torch
# import torch.nn as nn

# class network(nn.Module):
#     def __init__(self) -> None:
#         super(network, self).__init__()
#         self.struct = nn.Sequential(
#             nn.Linear(1, 1)
#         )
    
# net = network()
# print(net.parameters)

# import numpy as np
# import deepxde as dde

# x_lower = -5
# x_upper = 5
# t_lower = 0
# t_upper = np.pi / 2

# # 创建 2D 域（用于绘图和输入）
# x = np.linspace(x_lower, x_upper, 256)
# t = np.linspace(t_lower, t_upper, 201)
# X, T = np.meshgrid(x, t)

# # 整个域变平
# X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

# x = 1

# x = np.asarray([[1, 2], [3, 4], [5, 6]])
# print(x[:,0:1])
# print(np.shape(x[:, 0:1]))

# layer = [2] + [32] * 3 + [1]
# print(layer)

# x_data = np.linspace(-1, 1, num=100)
# t_data = np.linspace(0, 1, num=100)
# test_x, test_y = np.meshgrid(x_data, t_data)
# test_domain = np.vstack((np.ravel(test_x), np.ravel(test_y))).T
# t = 1

# def top_wall(x, on_boundary):
#     return on_boundary and np.isclose(x[1], 0.5)

# def zero_velocity_top_wall(x):
#     """Zero velocity"""
#     return np.zeros_like(top_wall)

# x = [[1, 2], [3, 4], [5, 6]]
# print(np.zeros_like(0))
# print(zero_velocity_top_wall(x))

# import torch

# x=torch.rand(2,2,requires_grad=True)
# print(x)
# y=2*x+2
# print(y)
# z=torch.sum(y)
# print(z)

# print(torch.autograd.grad(z,y)[0])
# dzdy = torch.autograd.grad(z,y)[0]
# print(torch.autograd.grad(y, x, grad_outputs=dzdy)[0])

# import tensorflow as tf
# print(tf.__version__)
import torch

def f(x, y, m=1, n=1, c=0.1, k=10):
    term1 = (4 * torch.pi**2 * m**2 * c * torch.sin(2 * torch.pi * m * x) +
             (2 * k**2 * torch.sinh(k * x) / torch.cosh(k * x)**3)) * torch.sin(2 * torch.pi * n * y)
    term2 = 4 * torch.pi**2 * n**2 * (c * torch.sin(2 * torch.pi * m * x) + torch.tanh(k * x)) * torch.sin(2 * torch.pi * n * y)
    return -(term1 + term2)

def x_line(n=10000):
    x = torch.linspace(-1, 1, n)
    y = torch.rand(1) * 2 - 1
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    x = xx.reshape(-1, 1)
    y = yy.reshape(-1, 1)
    condition = f(x, y)
    return x.requires_grad_(True), y.requires_grad_(True), condition

x, y, c = x_line()
print(x)
print(y)
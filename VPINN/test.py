import matplotlib.pyplot as plt
import torch
import os, sys
from VPINN2d import VPINN
import numpy as np
os.chdir(sys.path[0])

# define the pde and boundary condition
def u(x, y):
    return (0.1 * torch.sin(2 * torch.pi * x) + torch.tanh(10 * x)) * torch.sin(2 * torch.pi * y)

def f(x, y, m=1, n=1, c=0.1, k=10):
    term1 = (4 * torch.pi**2 * m**2 * c * torch.sin(2 * torch.pi * m * x) +
             (2 * k**2 * torch.sinh(k * x) / torch.cosh(k * x)**3)) * torch.sin(2 * torch.pi * n * y)
    term2 = 4 * torch.pi**2 * n**2 * (c * torch.sin(2 * torch.pi * m * x) + torch.tanh(k * x)) * torch.sin(2 * torch.pi * n * y)
    return -(term1 + term2)

def pde(x, y, u):
    return VPINN.LAPLACE_TERM(5 * (5 + (VPINN.laplace(x, y, u)))) - 5 * f(x, y) - 25

def transform(x):
    return x

# boundary condition
def bc(boundary_num, device='cpu'):
    xs = []
    ys = []
    x1, y1, x2, y2 = (-1, -1, 1, 1)
    x_r = torch.linspace(x2, x2, boundary_num).reshape(-1, 1).to(device).requires_grad_(True)
    y_r = torch.linspace(y1, y2, boundary_num).reshape(-1, 1).to(device).requires_grad_(True)
            
    x_u = torch.linspace(x1, x2, boundary_num).reshape(-1, 1).to(device).requires_grad_(True)
    y_u = torch.linspace(y2, y2, boundary_num).reshape(-1, 1).to(device).requires_grad_(True)
                
    x_l = torch.linspace(x1, x1, boundary_num).reshape(-1, 1).to(device).requires_grad_(True)
    y_l = torch.linspace(y1, y2, boundary_num).reshape(-1, 1).to(device).requires_grad_(True)
                
    x_d = torch.linspace(x1, x2, boundary_num).reshape(-1, 1).to(device).requires_grad_(True)
    y_d = torch.linspace(y1, y1, boundary_num).reshape(-1, 1).to(device).requires_grad_(True)
                
    xs.extend([x_r, x_u, x_l, x_d])
    ys.extend([y_r, y_u, y_l, y_d])
    boundary_xs = torch.cat(xs, dim=0)
    boundary_ys = torch.cat(ys, dim=0)
    boundary_us = u(boundary_xs, boundary_ys)
    return (boundary_xs, boundary_ys, boundary_us)


device = 'cpu'
# train the model
vpinn = VPINN([2, 15, 15, 15, 1], pde, bc(80, device=device), Q=30, grid_num=6, test_fcn_num=5, 
            device=device, load=None)
net = vpinn.train("Poisson", epoch_num=10000, coef=10)
net = vpinn.train(None, epoch_num=0, coef=10)


# verify
# data = np.loadtxt('poisson_boltzmann2d.dat', skiprows=9)

# # get x、y、u of solution
# x = data[:, 0]
# y = data[:, 1]
# u = data[:, 2]

# x_tensor = torch.from_numpy(x).reshape(-1, 1).type(torch.float).to(device)
# y_tensor = torch.from_numpy(y).reshape(-1, 1).type(torch.float).to(device)
# u_tensor = torch.from_numpy(u).reshape(-1, 1).type(torch.float).to(device)

# prediction = net(torch.cat([x_tensor, y_tensor], dim=1))
# print(f'median error={torch.median(torch.abs(u_tensor - prediction))}')
# print(f'relative error={torch.norm(u_tensor - prediction) / torch.norm(u_tensor) * 100:.2f}%')

# plot and verify
xc = torch.linspace(-1, 1, 500)
xx, yy = torch.meshgrid(xc, xc, indexing='ij')
xx = xx.reshape(-1, 1).to(device)
yy = yy.reshape(-1, 1).to(device)
xy = torch.cat([xx, yy], dim=1)
prediction = net(xy).to('cpu')
xx = xx.to('cpu')
yy = yy.to('cpu')

prediction = prediction.reshape(500, 500)
prediction = prediction.transpose(0, 1)
plt.imshow(prediction.detach().numpy(), cmap='hot', origin='lower')
plt.colorbar()
plt.savefig('prediction')
plt.show()

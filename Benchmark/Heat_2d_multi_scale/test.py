import torch
import numpy as np
import sys, os
import copy
from pyinstrument import Profiler
from matplotlib.tri import Triangulation
from pathlib import Path

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
os.chdir(sys.path[0])
from src.VPINN3d import VPINN

def u(x, y, z):
    return x ** 2 + y ** 2 + z ** 2

def pde(x, y, t, u):
    return VPINN.gradients(u, t, 1) - 1 / ((500 * torch.pi) ** 2) * VPINN.gradients(u, x, 2) - 1 / (torch.pi ** 2) * VPINN.gradients(u, y, 2)

def bc(boundary_num=10):
    x = torch.linspace(0, 1, boundary_num)
    y = torch.linspace(0, 1, boundary_num)
    t = torch.linspace(0, 5, boundary_num)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    xt, tt = torch.meshgrid(x, t, indexing='ij')
    
    bc_xs = [xx, xt, xt, torch.full_like(xt, 0), torch.full_like(xt, 1)]
    bc_ys = [yy, torch.full_like(xx, 0), torch.full_like(xt, 1), xt, xt]
    bc_ts = [torch.zeros_like(xx), tt, tt, tt, tt]
    bc_us = [torch.sin(20 * torch.pi * xx) * torch.sin(torch.pi * yy), torch.full_like(xt, 0), torch.full_like(xt, 0), torch.full_like(xt, 0), torch.full_like(xt, 0)]
    
    bc_xs = torch.cat(bc_xs, dim=0).reshape(-1, 1)
    bc_ys = torch.cat(bc_ys, dim=0).reshape(-1, 1)
    bc_ts = torch.cat(bc_ts, dim=0).reshape(-1, 1)
    bc_us = torch.cat(bc_us, dim=0).reshape(-1, 1)
    return (bc_xs, bc_ys, bc_ts, bc_us)

device = 'cuda'
vpinn = VPINN([3, 10, 10, 10, 1],pde, bc(10), area=[0, 1, 0, 1, 0, 5], Q=10, grid_num=4, test_fcn_num=5, 
            device=device, load=None)

# profiler=Profiler()
# profiler.start()

net = vpinn.train('Poisson3d', epoch_num=10000, coef=5)

# profiler.stop()
# profiler.print()
# net = vpinn.train(None, epoch_num=0, coef=0.1)

################################################################################
# verify
data = np.loadtxt('heat_multiscale_lesspoints.dat', skiprows=9)

# get x、y、u of solution
x_ = data[:, 0]
y_ = data[:, 1]
x = []
y = []
t = []
u = []
for i in range(26):
    x.append(copy.deepcopy(x_))
    y.append(copy.deepcopy(y_))
    t.append([i * 0.2 for _ in range(len(x_))])
    u.append(data[:, i + 2])

x = np.concatenate(x)
y = np.concatenate(y)
t = np.concatenate(t)
u = np.concatenate(u)
tri = Triangulation(x, t)

xx = torch.from_numpy(x).reshape(-1, 1).type(torch.float)
yy = torch.from_numpy(y).reshape(-1, 1).type(torch.float)
tt = torch.from_numpy(t).reshape(-1, 1).type(torch.float)
uu = torch.from_numpy(u).reshape(-1, 1).type(torch.float)

net.cpu()
# x = torch.linspace(-1, 1, 50)
# y = x
# z = x
# xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
# xx = xx.reshape(-1, 1)
# yy = yy.reshape(-1, 1)
# zz = zz.reshape(-1, 1)

prediction = net(torch.cat([xx, yy, tt], dim=1))
# solution = u(xx, yy, zz)
print(f'relative error={torch.norm(prediction - uu):.2f}/{torch.norm(uu):.2f}={torch.norm(prediction - uu) / torch.norm(uu) * 100:.2f}%')
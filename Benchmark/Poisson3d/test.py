import torch
import numpy as np
import sys, os
# import pyinstrument
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from pathlib import Path

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
os.chdir(sys.path[0])
from src.VPINN3d import VPINN

def u(x, y, z):
    return x ** 2 + y ** 2 + z ** 2

def pde(x, y, z, u):
    return VPINN.gradients(u, x, 2) + VPINN.gradients(u, y, 2) + VPINN.gradients(u, z, 2) - 6

def bc(boundary_num=10):
    x = torch.linspace(-1, 1, boundary_num)
    y = torch.linspace(-1, 1, boundary_num)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    bc_xs = [xx, xx, xx, xx, torch.full_like(xx, 1), torch.full_like(yy, 1)]
    bc_ys = [yy, yy, torch.full_like(xx, 1), torch.full_like(xx, -1), xx, xx]
    bc_zs = [torch.full_like(xx, 1), torch.full_like(xx, -1), yy, yy, yy, yy]
    bc_xs = torch.cat(bc_xs, dim=0).reshape(-1, 1)
    bc_ys = torch.cat(bc_ys, dim=0).reshape(-1, 1)
    bc_zs = torch.cat(bc_zs, dim=0).reshape(-1, 1)
    bc_us = u(bc_xs, bc_ys, bc_zs)
    return (bc_xs, bc_ys, bc_zs, bc_us)

device = 'cpu'
vpinn = VPINN([3, 10, 10, 10, 1],pde, bc(10), area=[-1, -1, 1, 1, -1, 1], Q=10, grid_num=4, test_fcn_num=5, 
            device=device, load=None)
net = vpinn.train('Poisson3d', epoch_num=10000, coef=1)

################################################################################
x = torch.linspace(-1, 1, 50)
y = x
z = x
xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
xx = xx.reshape(-1, 1)
yy = yy.reshape(-1, 1)
zz = zz.reshape(-1, 1)

prediction = net(torch.cat([xx, yy, zz], dim=1))
solution = u(xx, yy, zz)
print(f'relative error={torch.norm(prediction - solution):.2f}/{torch.norm(solution):.2f}={torch.norm(prediction - solution) / torch.norm(solution) * 100:.2f}%')
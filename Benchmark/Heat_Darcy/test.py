import torch
import numpy as np
import sys, os
# import pyinstrument
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import interpolate

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
os.chdir(sys.path[0])
from src.VPINN3d import VPINN

#############################################################################################
# define the pde
def f(x, y, t):
    A = 10
    m1 = 1
    m2 = 5
    m3 = 1
    sin = torch.sin
    pi = torch.pi
    return A * sin(m1 * pi * x) * sin(m2 * pi * y) * sin(m3 * pi * t)

# VPINN.laplace represents the result of laplace operator applied to u with Green Theorem
# When VPINN.laplace is used, you are expected to wrap the monomial with VPINN.LAPLACE_TERM() to distinguish
# VPINN.LAPLACE_TERM can only contain a monomial of VPINN.laplace, polynomial is not allowed

def coef(x, y, t):
    heat_2d_coef = np.loadtxt("heat_2d_coef_256.dat")
    X = torch.cat([x, y, t], dim=1)
    return torch.Tensor(
                interpolate.griddata(heat_2d_coef[:, 0:2], heat_2d_coef[:, 2], X.detach().cpu().numpy()[:, 0:2], method="nearest")
            )
def pde(x, y, t, u, device='cuda'):    
    return VPINN.gradients(u, t, 1) - coef(x, y, t).to(device).reshape(-1, 1) * (VPINN.gradients(u, x, 2) + VPINN.gradients(u, y, 2)) - f(x, y, t)

# this pde doesn't use the green theorem to simplify the equation
# def pde(x, y, u):    
    # return VPINN.gradients(u, x, 2) + VPINN.gradients(u, y, 2) - f(x, y)

#############################################################################################
# boundary condition
def bc(boundary_num):
    x = torch.linspace(0, 1, boundary_num)
    y = torch.linspace(0, 1, boundary_num)
    t = torch.linspace(0, 5, boundary_num)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    xt, tt = torch.meshgrid(x, t, indexing='ij')
    
    xx = xx.reshape(-1, 1)
    yy = yy.reshape(-1, 1)
    xt = xt.reshape(-1, 1)
    tt = tt.reshape(-1, 1)
    
    bc_xs = [xx, xt, xt, torch.full_like(xt, 0), torch.full_like(xt, 1)]
    bc_ys = [yy, torch.full_like(xx, 0), torch.full_like(xt, 1), xt, xt]
    bc_ts = [torch.zeros_like(xx), tt, tt, tt, tt]
    bc_us = [torch.full_like(xt, 0), torch.full_like(xt, 0), torch.full_like(xt, 0), torch.full_like(xt, 0), torch.full_like(xt, 0)]
    
    bc_xs = torch.cat(bc_xs, dim=0).reshape(-1, 1)
    bc_ys = torch.cat(bc_ys, dim=0).reshape(-1, 1)
    bc_ts = torch.cat(bc_ts, dim=0).reshape(-1, 1)
    bc_us = torch.cat(bc_us, dim=0).reshape(-1, 1)
    return (bc_xs, bc_ys, bc_ts, bc_us)

#############################################################################################
# train the model
def heat_darcy():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vpinn = VPINN([3, 10, 10, 10, 1],pde, bc(100), area=[0, 1, 0, 1, 0, 5], Q=10, grid_num=4, test_fcn_num=5, 
                device=device, load=None)

    net = vpinn.train('heat_darcy', epoch_num=10000, coef=10)
    net.cpu()
    #############################################################################################
    # plot and verify
    

if __name__ == "__main__":
    heat_darcy()
import torch

sin = torch.sin
cos = torch.cos
pi = torch.pi
# define the pde in the form of N(u,\lambda)=f

def u(x, y):
    return sin(pi / 2 * (x + 1)) * sin(pi / 2 * (y + 1)) 

def f(x, y):
    return -pi ** 2 / 2 * u(x, y)

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
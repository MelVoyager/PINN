import torch

def u(x, y):
    return (0.1 * torch.sin(2 * torch.pi * x) + torch.tanh(10 * x)) * torch.sin(2 * torch.pi * y)

def f(x, y, m=1, n=1, c=0.1, k=10):
    term1 = (4 * torch.pi**2 * m**2 * c * torch.sin(2 * torch.pi * m * x) +
             (2 * k**2 * torch.sinh(k * x) / torch.cosh(k * x)**3)) * torch.sin(2 * torch.pi * n * y)
    term2 = 4 * torch.pi**2 * n**2 * (c * torch.sin(2 * torch.pi * m * x) + torch.tanh(k * x)) * torch.sin(2 * torch.pi * n * y)
    return -(term1 + term2)

        
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
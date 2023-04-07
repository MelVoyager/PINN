import torch
import copy

def f(x, t):
    return torch.zeros_like(x)

def bc(boundary_num, device='cpu'):
    xs = []
    ts = []
    us = []
    x = torch.linspace(-1, 1, boundary_num).reshape(-1, 1)
    t = torch.zeros_like(x)
    u = -torch.sin(torch.pi * x)
    xs.append(x)
    ts.append(t)
    us.append(u)
    
    t = torch.linspace(0, 1, boundary_num).reshape(-1, 1)
    x1 = torch.full_like(t, -1)
    x2 = torch.full_like(t, 1)
    u = torch.zeros_like(t)
    
    xs.append(x1)
    ts.append(copy.deepcopy(t))
    us.append(copy.deepcopy(u))
    
    xs.append(x2)
    ts.append(copy.deepcopy(t))
    us.append(copy.deepcopy(u))
    
    for i in range(len(xs)):
        xs[i].to(device).requires_grad_(True)
        ts[i].to(device).requires_grad_(True)
        us[i].to(device).requires_grad_(True)
    
    boundary_xs = torch.cat(xs, dim=0).to(device)
    boundary_ts = torch.cat(ts, dim=0).to(device)
    boundary_us = torch.cat(us, dim=0).to(device)
    return (boundary_xs, boundary_ts, boundary_us)
    
    
    
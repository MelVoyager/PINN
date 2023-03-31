import torch

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
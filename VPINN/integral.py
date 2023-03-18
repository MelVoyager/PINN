import torch

def integral(func, l=-1, r=1, density=10000):
    x = torch.linspace(l, r, density)
    return torch.sum(func(x) * (r - l) / density)

def f(x):
    return torch.sin(x)

print(integral(f, -torch.pi / 2, torch.pi / 2))
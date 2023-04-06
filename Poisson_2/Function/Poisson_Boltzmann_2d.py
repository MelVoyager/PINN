import torch

sin = torch.sin
cos = torch.cos
pi = torch.pi
def f(x, y):
    return 10 * (17 + x ** 2 + y ** 2) * sin(pi * x) * sin(4 * pi * y)

def u(x, y):
    return torch.zeros_like(x)
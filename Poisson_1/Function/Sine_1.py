import torch

sin = torch.sin
cos = torch.cos
pi = torch.pi
# define the pde in the form of N(u,\lambda)=f

def u(x, y):
    return sin(pi / 2 * (x + 1)) * sin(pi / 2 * (y + 1)) 

def f(x, y):
    return -pi ** 2 / 2 * u(x, y)
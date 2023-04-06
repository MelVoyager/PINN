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

def gradients(u, x, order=1):
    if order == 1:
        return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                   create_graph=True,
                                   only_inputs=True, )[0]
    else:
        return gradients(gradients(u, x), x, order=order - 1)
    
def rand_in_interval(size, l=-1, r=1):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return (torch.rand(size) * (r - l) + torch.full(size, l)).to(device)

def index2frame(index, grid_num):
    i = index // grid_num
    j = index % grid_num
    grid_len = 2 / grid_num
    x1 = -1 + i * grid_len
    y1 = -1 + j * grid_len
    x2 = -1 + (i + 1) * grid_len
    y2 = -1 + (j + 1) * grid_len
    return x1, y1, x2, y2

def gradient_u_2order(x, y):
    u_xx = (-3.94784 * torch.sin(2 * torch.pi * y) *(torch.sin(2 * torch.pi * x) + 50.6606 * torch.tanh(10 * x) * (1 / torch.cosh(10 * x))**2))
    u_yy = (-3.94784 * torch.sin(2 * torch.pi * y) *(torch.sin(2 * torch.pi * x) + 10 * torch.tanh(10 * x)))
    return torch.cat([u_xx, u_yy], dim=1)


        
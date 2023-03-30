import torch 

def legendre(n, x):
    if n == 0:
        return torch.ones_like(x)
    elif n == 1:
        return x
    else:
        return ((2 * n - 1) * x * legendre(n - 1, x) - (n - 1) * legendre(n - 2, x)) / n

def legendre_derivative(n, x):
    if n == 0:
        return torch.zeros_like(x)
    elif n == 1:
        return torch.ones_like(x)
    else:
        return (n * (legendre(n - 1, x) - x * legendre(n, x))) / (1 - x**2)
    
def u(k, n, x):
    if k == 0:
        return legendre(n, x)
    elif k == 1:
        return legendre_derivative(n, x)
    else:
        raise ValueError("k must be 0 or 1")

def v(k, n, x, y):
    if k == 0:
        return (legendre(n + 1, x) - legendre(n - 1, x)) * (legendre(n + 1, y) - legendre(n - 1, y))
    
    elif k == 1:
        return torch.cat([(legendre_derivative(n + 1, x) - legendre_derivative(n - 1, x)) * (legendre(n + 1, y) - legendre(n - 1, y)), (legendre(n + 1, x) - legendre(n - 1, x)) * (legendre_derivative(n + 1, y) - legendre_derivative(n - 1, y))], dim=1)
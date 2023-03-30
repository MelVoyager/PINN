import torch
from scipy.integrate import dblquad
from Lengendre import Lengendre
import matplotlib.pyplot as plt
def u(x, y):
    return (0.1 * torch.sin(2 * torch.pi * x) + torch.tanh(10 * x)) * torch.sin(2 * torch.pi * y)

def f(x, y, m=1, n=1, c=0.1, k=10):
    term1 = (4 * torch.pi**2 * m**2 * c * torch.sin(2 * torch.pi * m * x) +
             (2 * k**2 * torch.sinh(k * x) / torch.cosh(k * x)**3)) * torch.sin(2 * torch.pi * n * y)
    term2 = 4 * torch.pi**2 * n**2 * (c * torch.sin(2 * torch.pi * m * x) + torch.tanh(k * x)) * torch.sin(2 * torch.pi * n * y)
    return -(term1 + term2)

def v(x, y):
    return 2.5 * (x ** 3 - x) * 2.5 * (y ** 3 - y)

def u_gradient(x, y):
    u_val = u(x, y)
    u_val.backward(torch.ones_like(x))
    
    # grad_u = torch.autograd.grad(u_val, [x, y])
    return torch.cat([x.grad, y.grad], dim=1)

def v_gradient(x, y):
    v_val = v(x, y)
    v_val.backward(torch.ones_like(x))
    return torch.cat([x.grad, y.grad], dim=1)

def gradient_u(x, y):
    dx = (0.2 * torch.pi * torch.cos(2*torch.pi*x) + 10 * (1 / torch.cosh(10*x))**2) * torch.sin(2 * torch.pi * y)
    dy = 2 * torch.pi * (0.1 * torch.sin(2*torch.pi*x) + torch.tanh(10*x)) * torch.cos(2*torch.pi*y)
    return torch.cat([dx, dy], dim=1)

lengendre = Lengendre()

def gradient_v(x, y, k=1):
    return torch.cat([lengendre.v_prime(x, k) * lengendre.v(y, k), lengendre.v(x, k) * lengendre.v_prime(y, k)], dim=1)

N = 100
eps = torch.rand(1).item() * 0.1
x = torch.linspace(-1, 1 - eps, N) + eps
y = torch.linspace(-1, 1 - eps, N) + eps
xx, yy = torch.meshgrid(x, y, indexing="ij")
xx = xx.reshape(-1, 1).requires_grad_(True)
yy = yy.reshape(-1, 1).requires_grad_(True)

print(torch.sum(u_gradient(xx, yy) * v_gradient(xx, yy)))
print(torch.sum(f(xx, yy) * v(xx, yy)))

def wrapper(x, y):
    x = torch.tensor(x)
    y = torch.tensor(y)
    return (f(x, y) * v(x, y)).item()

def wrapper2(x, y):
    x = torch.tensor(x).reshape(-1, 1)
    y = torch.tensor(y).reshape(-1, 1)
    return torch.sum(gradient_u(x, y) * gradient_v(x, y, k=2)).item()
    # dx = 0.2 * torch.pi * torch.cos(2*torch.pi*x) + 10 * (1 / torch.cosh(10*x))**2
    # dy = 2 * torch.pi * (0.1 * torch.sin(2*torch.pi*x) + torch.tanh(10*x)) * torch.cos(2*torch.pi*y)


'''
def integral():
    return dblquad(wrapper2, -1, 1, lambda x: -1, lambda x: 1)

# print(wrapper2(0.5, -0.5))
result, error = integral()
print("积分结果为:", result)
'''
# plt.imshow((gradient_v(xx, yy, k=2)[:,1]).reshape(N, N).detach().numpy())
plt.imshow((v(xx, yy)).reshape(N, N).detach().numpy())
# plt.imshow((v_gradient(xx, yy)[:,0]).reshape(N, N).detach().numpy())
# plt.imshow((torch.sum(gradient_u(xx, yy) * gradient_v(xx, yy, k=2),dim=1, keepdim=True)).reshape(N, N).detach().numpy())
# plt.imshow((f(xx, yy) * v(xx, yy)).reshape(N, N).detach().numpy())
plt.colorbar()
plt.show()


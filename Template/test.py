import matplotlib.pyplot as plt
import torch
from src.VPINN2d import VPINN

#############################################################################################
# define the pde
def u(x, y):
    return (0.1 * torch.sin(2 * torch.pi * x) + torch.tanh(10 * x)) * torch.sin(2 * torch.pi * y)

def f(x, y, m=1, n=1, c=0.1, k=10):
    term1 = (4 * torch.pi**2 * m**2 * c * torch.sin(2 * torch.pi * m * x) +
             (2 * k**2 * torch.sinh(k * x) / torch.cosh(k * x)**3)) * torch.sin(2 * torch.pi * n * y)
    term2 = 4 * torch.pi**2 * n**2 * (c * torch.sin(2 * torch.pi * m * x) + torch.tanh(k * x)) * torch.sin(2 * torch.pi * n * y)
    return -(term1 + term2)

# VPINN.laplace represents the result of laplace operator applied to u with Green Theorem
# When VPINN.laplace is used, you are expected to wrap the monomial with VPINN.LAPLACE_TERM() to distinguish
# VPINN.LAPLACE_TERM can only contain a monomial of VPINN.laplace, polynomial is not allowed

def pde(x, y, u):    
    return VPINN.LAPLACE_TERM(VPINN.laplace(x, y, u)) - f(x, y)

# this pde doesn't use the green theorem to simplify the equation
# def pde(x, y, u):    
    # return VPINN.gradients(u, x, 2) + VPINN.gradients(u, y, 2) - f(x, y)

#############################################################################################
# boundary condition
def bc(boundary_num):
    xs = []
    ys = []
    x1, y1, x2, y2 = (-1, -1, 1, 1)
    x_r = torch.linspace(x2, x2, boundary_num).reshape(-1, 1)
    y_r = torch.linspace(y1, y2, boundary_num).reshape(-1, 1)
            
    x_u = torch.linspace(x1, x2, boundary_num).reshape(-1, 1)
    y_u = torch.linspace(y2, y2, boundary_num).reshape(-1, 1)
                
    x_l = torch.linspace(x1, x1, boundary_num).reshape(-1, 1)
    y_l = torch.linspace(y1, y2, boundary_num).reshape(-1, 1)
                
    x_d = torch.linspace(x1, x2, boundary_num).reshape(-1, 1)
    y_d = torch.linspace(y1, y1, boundary_num).reshape(-1, 1)
                
    xs.extend([x_r, x_u, x_l, x_d])
    ys.extend([y_r, y_u, y_l, y_d])
    boundary_xs = torch.cat(xs, dim=0)
    boundary_ys = torch.cat(ys, dim=0)
    boundary_us = u(boundary_xs, boundary_ys)
    return (boundary_xs, boundary_ys, boundary_us)

#############################################################################################
# train the model
device = 'mps'
vpinn = VPINN([2, 15, 15, 15, 1], pde, bc(80), Q=10, grid_num=6, test_fcn_num=5, 
            device=device, load=None)


net = vpinn.train("Poisson", epoch_num=10000, coef=10)

#############################################################################################
# plot and verify
xc = torch.linspace(-1, 1, 500)
xx, yy = torch.meshgrid(xc, xc, indexing='ij')
xx = xx.reshape(-1, 1).to(device)
yy = yy.reshape(-1, 1).to(device)
xy = torch.cat([xx, yy], dim=1)
prediction = net(xy).to('cpu')
xx = xx.to('cpu')
yy = yy.to('cpu')

prediction = prediction.reshape(500, 500)
prediction = prediction.transpose(0, 1)
plt.imshow(prediction.detach().numpy(), cmap='hot', origin='lower')
plt.colorbar()
plt.savefig('prediction')
plt.show()


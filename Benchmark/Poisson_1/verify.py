import torch
from Utilities.Integral2d import quad_integral
from Utilities.lengendre import test_func, v
from Function.Sine_1 import u, f

def gradient_u(x, y):
        du_dx = 0.5 * torch.pi * torch.cos(0.5 * torch.pi * (x + 1)) * torch.sin(0.5 * torch.pi * (y + 1))
        du_dy = torch.sin(0.5 * torch.pi * (x + 1)) * 0.5 * torch.pi * torch.cos(0.5 * torch.pi * (y + 1))
        return du_dx, du_dy

test_func.init(5)
Q = 100
quad_integral.init(Q)
    

def index2frame(index, grid_num):
        i = index // grid_num
        j = index % grid_num
        grid_len = 2 / grid_num
        x1 = -1 + i * grid_len
        y1 = -1 + j * grid_len
        x2 = -1 + (i + 1) * grid_len
        y2 = -1 + (j + 1) * grid_len
        return x1, y1, x2, y2
    
grid_num = 4
xs = []
ys = []
x = quad_integral.XX
y = quad_integral.YY
for index in range(grid_num ** 2):
    x1, y1, x2, y2 = index2frame(index, grid_num)
    xx = (x.reshape(-1, 1).requires_grad_(True) + 1) / grid_num + x1
    yy = (y.reshape(-1, 1).requires_grad_(True) + 1) / grid_num + y1
    xs.append(xx)
    ys.append(yy)
xs = torch.cat(xs, dim=0).view(-1, 1)
ys = torch.cat(ys, dim=0).view(-1, 1)
grid_xs = xs
grid_ys = ys
            
def fWrapper(X, Y):
    result = f(grid_xs, grid_ys).view(grid_num ** 2, Q ** 2) * test_func.test_func(0, X, Y).view(1, Q ** 2)
    result = torch.sum(result, dim=0, keepdim=True)
    return result


def DeltaWrapper(X, Y):
    dx, dy = gradient_u(grid_xs, grid_ys)
    result = (torch.sum(torch.cat([dx, dy], dim=1).view(grid_num ** 2, Q ** 2, 2) * test_func.test_func(1, X, Y).view(1, Q ** 2, 2), dim=2, keepdim=True)).view(grid_num ** 2, Q ** 2)
    result = torch.sum(result, dim=0, keepdim=True)
    return result

print(quad_integral.integral(fWrapper))
print(quad_integral.integral(DeltaWrapper) * grid_num)

print(v(0,1,1,torch.full((1,), 0), torch.full((1,), -1)))
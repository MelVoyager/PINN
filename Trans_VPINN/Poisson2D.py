import torch
from tqdm import tqdm
from net_class import MLP
from basis import rand_in_interval, u, index2frame
from Integral2d import quad_integral
from lengendre import test_func
import basis
import json
import lengendre
import os, sys
import math
import cProfile
import pstats
import numpy as np

os.chdir(sys.path[0])
pi = torch.pi
sin = torch.sin
cos = torch.cos
test_num = 3
test_fcn = 5
grid_num = 8
N = 100

def output(tensor, name):
    tensor_str = str(tensor.detach().numpy())
    np.set_printoptions(threshold=np.inf, linewidth=np.inf, precision=6)
    with open(name, 'w') as f:
        f.write(tensor_str)

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = torch.device("mps") if torch.has_mps else "cpu"
print(f"Using {device} device")

def grid(n=N):
    eps = torch.rand(1).item() * 0.1
    x = (torch.linspace(-1, 1 - eps, n) + eps / 2).to(device)
    y = (torch.linspace(-1, 1, n) + eps / 2).to(device)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    x = xx.reshape(-1, 1)
    y = yy.reshape(-1, 1)
    # perm = torch.randperm(len(x))
    # x = x[perm]
    # y = y[perm]
    condition = basis.f(x, y)
    return x.requires_grad_(True), y.requires_grad_(True), condition

# (x1,y1)左下,(x2, y2)右上
def bc1(x1, y1, x2, y2, n=80):
    x = torch.linspace(x2, x2, n).reshape(-1, 1).to(device)
    y = torch.linspace(y1, y2, n).reshape(-1, 1).to(device)
    condition = u(x, y)
    return x.requires_grad_(True), y.requires_grad_(True), condition

def bc2(x1, y1, x2, y2, n=80):
    x = torch.linspace(x1, x2, n).reshape(-1, 1).to(device)
    y = torch.linspace(y2, y2, n).reshape(-1, 1).to(device)
    condition = u(x, y)
    return x.requires_grad_(True), y.requires_grad_(True), condition

def bc3(x1, y1, x2, y2, n=80):
    x = torch.linspace(x1, x1, n).reshape(-1, 1).to(device)
    y = torch.linspace(y1, y2, n).reshape(-1, 1).to(device)
    condition = u(x, y)
    return x.requires_grad_(True), y.requires_grad_(True), condition

def bc4(x1, y1, x2, y2, n=80):
    x = torch.linspace(x1, x2, n).reshape(-1, 1).to(device)
    y = torch.linspace(y1, y1, n).reshape(-1, 1).to(device)
    condition = u(x, y)
    return x.requires_grad_(True), y.requires_grad_(True), condition

# def bc1(n=1000):
#     x = rand_in_interval((n, 1), l=1, r=1)
#     y = rand_in_interval((n, 1))
#     condition = basis.u(x, y)
#     return x.requires_grad_(True), y.requires_grad_(True), condition

# def bc2(n=1000):
#     x = rand_in_interval((n, 1))
#     y = rand_in_interval((n, 1), l=1)
#     condition = basis.u(x, y)
#     return x.requires_grad_(True), y.requires_grad_(True), condition

# def bc3(n=1000):
#     x = rand_in_interval((n, 1), r=-1)
#     y = rand_in_interval((n, 1))
#     condition = basis.u(x, y)
#     return x.requires_grad_(True), y.requires_grad_(True), condition

# def bc4(n=1000):
#     x = rand_in_interval((n, 1))
#     y = rand_in_interval((n, 1), r=-1)
#     condition = basis.u(x, y)
#     return x.requires_grad_(True), y.requires_grad_(True), condition

loss = torch.nn.MSELoss()

loss1 = [[] for _ in range(test_num)]
loss2 = [[] for _ in range(test_num)]
loss3 = [[] for _ in range(test_num)]

f = torch.tensor(0)
ext_laplace_u = torch.tensor(0)
laplace_u = torch.tensor(0)
def LaplaceWrapper(x, y, index=None, k1=1, k2=1):
    x_tot = []
    y_tot = []
    for index in range(grid_num ** 2):
        x1, y1, x2, y2 = index2frame(index, grid_num)
        xx = (x.clone().detach().reshape(-1, 1).requires_grad_(True) + 1) / grid_num + x1
        yy = (y.clone().detach().reshape(-1, 1).requires_grad_(True) + 1) / grid_num + y1
        x_tot.append(xx)
        y_tot.append(yy)
    x_tot = torch.cat(x_tot, dim=0).view(-1, 1)
    y_tot = torch.cat(y_tot, dim=0).view(-1, 1)
    
    u = net(torch.cat([x_tot, y_tot], dim=1))
    d2x = basis.gradients(u, x_tot, 2)
    d2y = basis.gradients(u, y_tot, 2)
    global ext_laplace_u
    global laplace_u
    laplace_u = d2x + d2y
    # ext_laplace_u = torch.sum(basis.gradient_u_2order(x_tot, y_tot), dim=1, keepdim=True)
    result = laplace_u.view(grid_num ** 2, 100) * (test_func.test_func(0, x, y).squeeze(-1).unsqueeze(0)).expand(grid_num ** 2, 100)
    return result

def ext_LaplaceWrapper(x, y, index=None, k1=1, k2=1):
    x_tot = []
    y_tot = []
    for index in range(grid_num ** 2):
        x1, y1, x2, y2 = index2frame(index, grid_num)
        xx = (x.clone().detach().reshape(-1, 1).requires_grad_(True) + 1) / grid_num + x1
        yy = (y.clone().detach().reshape(-1, 1).requires_grad_(True) + 1) / grid_num + y1
        x_tot.append(xx)
        y_tot.append(yy)
    x_tot = torch.cat(x_tot, dim=0).view(-1, 1)
    y_tot = torch.cat(y_tot, dim=0).view(-1, 1)
    
    u = net(torch.cat([x_tot, y_tot], dim=1))
    d2x = basis.gradients(u, x_tot, 2)
    d2y = basis.gradients(u, y_tot, 2)
    global ext_laplace_u
    global laplace_u
    laplace_u = d2x + d2y
    ext_laplace_u = torch.sum(basis.gradient_u_2order(x_tot, y_tot), dim=1, keepdim=True)
    result = ext_laplace_u.view(grid_num ** 2, 100) * (test_func.test_func(0, x, y).squeeze(-1).unsqueeze(0)).expand(grid_num ** 2, 100)
    return result

def fWrapper(x, y, index=None, k1=1, k2=1):
    x_tot = []
    y_tot = []
    for index in range(grid_num ** 2):
        x1, y1, x2, y2 = index2frame(index, grid_num)
        xx = (x.clone().detach().reshape(-1, 1).requires_grad_(True) + 1) / grid_num + x1
        yy = (y.clone().detach().reshape(-1, 1).requires_grad_(True) + 1) / grid_num + y1
        x_tot.append(xx)
        y_tot.append(yy)
    
    x_tot = torch.cat(x_tot, dim=0).view(-1, 1)
    y_tot = torch.cat(y_tot, dim=0).view(-1, 1)
    # print(x)
    # print(xx)
    global f
    f = basis.f(x_tot, y_tot)
    # print(f.T)
    result = f.view(grid_num ** 2, 100) * test_func.test_func(0, x, y).squeeze(-1).unsqueeze(0).expand(grid_num ** 2, 100)
    return result

def loss_interior_1(net, index=None, k1=1, k2=1):
    int1 = quad_integral.integral(LaplaceWrapper, k1, k2, index) * ((1 /grid_num) ** 2)
    int2 = quad_integral.integral(fWrapper, k1, k2, index) * ((1 /grid_num) ** 2)
    # int3 = quad_integral.integral(ext_LaplaceWrapper, k1, k2, index) * ((1 /grid_num) ** 2)
    err1 = torch.abs(f - laplace_u)
    err2 = torch.abs(f - ext_laplace_u)
    if k1 - 1 < test_num:
        loss1[k1-1].append(loss(int1, int2).item())
    
    # print(loss(int1, int2))
    return loss(int1, int2)

def gradient_u(x, y):
    dx = (0.2 * torch.pi * torch.cos(2*torch.pi*x) + 10 * (1 / torch.cosh(10*x))**2) * torch.sin(2 * torch.pi * y)
    dy = 2 * torch.pi * (0.1 * torch.sin(2*torch.pi*x) + torch.tanh(10*x)) * torch.cos(2*torch.pi*y)
    return torch.cat([dx, dy], dim=1)

def DeltaWrapper(x, y, index, k1=1, k2=1):
    x_tot = []
    y_tot = []
    for index in range(grid_num ** 2):
        x1, y1, x2, y2 = index2frame(index, grid_num)
        xx = (x.clone().detach().reshape(-1, 1).requires_grad_(True) + 1) / grid_num + x1
        yy = (y.clone().detach().reshape(-1, 1).requires_grad_(True) + 1) / grid_num + y1
        x_tot.append(xx)
        y_tot.append(yy)
    
    x_tot = torch.cat(x_tot, dim=0).view(-1, 1)
    y_tot = torch.cat(y_tot, dim=0).view(-1, 1)
    
    u = net(torch.cat([x_tot, y_tot], dim=1))
    dx = basis.gradients(u, x_tot, 1)
    dy = basis.gradients(u, y_tot, 1)
    du = torch.cat([dx, dy], dim=1) * grid_num
    # output(du,'1')
    # result =  torch.bmm(f.view(100, grid_num ** 2, 1), test_func.test_func(0, x, y).view(100, 1, 1))
    # return torch.sum(grid_num * du * lengendre.v(1, k1, k2, x, y),dim=1, keepdim=True)
    n = grid_num * grid_num
    du = du.view(n, 100, 2)
    # output(du,'2')
    # output(test_func.test_func(1, x, y), '3')
    dv = (test_func.test_func(1, x, y)).unsqueeze(0).repeat(n, 1, 1)
    # output(dv, '4')
    result = (du * dv).sum(dim=-1)
    # output(result, '5')
    # output(result.reshape(100, 16, 1), '6')
    return result

def loss_interior_2(net, index=None, k1=1, k2=1):
    
    int1 = torch.abs(quad_integral.integral(DeltaWrapper, k1, k2, index)) * ((1 /grid_num) ** 2)
    int2 = torch.abs(quad_integral.integral(fWrapper, k1, k2, index)) * ((1 /grid_num) ** 2)
    # int3 = quad_integral()
    # print(int1, int2, int3)
    # if k - 1 < test_num:
    #     loss2[k-1].append(loss(int1, int2).item())
    # print(loss(int1, int2))
    return loss(int1, int2)

# def loss_interior_3(net, k=1):
#     x, y, condition = x_line()
#     output = net(torch.cat([x, y], dim=1))
#     y_grad_2order = gradients(output, y, 2)
#     int1 = integral(lengendre.v_2prime(x, k), multipier=output) \
#         - (net(torch.cat([x, y], dim=1)[-1]) * lengendre.v_prime(1, k) - net(torch.cat([x, y], dim=1)[0]) * lengendre.v_prime(-1, k)) \
#         + integral(lengendre.v(x, k), multipier=(y_grad_2order))
#     int1 = torch.sum(int1)
#     int2 = integral(lengendre.v(x, k), multipier=condition)
    
#     loss3[k-1].append(loss(int1, int2).item())
#     return loss(int1, int2)

def loss_bc1(net):
    x_tot = []
    y_tot = []
    condition_tot = []
    for index in range(grid_num ** 2):
        x1, y1, x2, y2 = index2frame(index, grid_num)
        x, y, condition = bc1(x1, y1, x2, y2)
        x_tot.append(x)
        y_tot.append(y)
        condition_tot.append(condition)
        
    x_tot = torch.cat(x_tot, dim=0).view(-1, 1)
    y_tot = torch.cat(y_tot, dim=0).view(-1, 1)
    
    output = net(torch.cat([x_tot, y_tot], dim=1))
    condition_tot = torch.cat(condition_tot, dim=0).reshape(-1, 1)
    return loss(output, condition_tot)

def loss_bc2(net):
    x_tot = []
    y_tot = []
    condition_tot = []
    for index in range(grid_num ** 2):
        x1, y1, x2, y2 = index2frame(index, grid_num)
        x, y, condition = bc2(x1, y1, x2, y2)
        x_tot.append(x)
        y_tot.append(y)
        condition_tot.append(condition)
        
    x_tot = torch.cat(x_tot, dim=0).view(-1, 1)
    y_tot = torch.cat(y_tot, dim=0).view(-1, 1)
    
    output = net(torch.cat([x_tot, y_tot], dim=1))
    condition_tot = torch.cat(condition_tot, dim=0).reshape(-1, 1)
    return loss(output, condition_tot)

def loss_bc3(net):
    x_tot = []
    y_tot = []
    condition_tot = []
    for index in range(grid_num ** 2):
        x1, y1, x2, y2 = index2frame(index, grid_num)
        x, y, condition = bc3(x1, y1, x2, y2)
        x_tot.append(x)
        y_tot.append(y)
        condition_tot.append(condition)
        
    x_tot = torch.cat(x_tot, dim=0).view(-1, 1)
    y_tot = torch.cat(y_tot, dim=0).view(-1, 1)
    
    output = net(torch.cat([x_tot, y_tot], dim=1))
    condition_tot = torch.cat(condition_tot, dim=0).reshape(-1, 1)
    return loss(output, condition_tot)

def loss_bc4(net):
    x_tot = []
    y_tot = []
    condition_tot = []
    for index in range(grid_num ** 2):
        x1, y1, x2, y2 = index2frame(index, grid_num)
        x, y, condition = bc4(x1, y1, x2, y2)
        x_tot.append(x)
        y_tot.append(y)
        condition_tot.append(condition)
        
    x_tot = torch.cat(x_tot, dim=0).view(-1, 1)
    y_tot = torch.cat(y_tot, dim=0).view(-1, 1)
    
    output = net(torch.cat([x_tot, y_tot], dim=1))
    condition_tot = torch.cat(condition_tot, dim=0).reshape(-1, 1)
    return loss(output, condition_tot)

# def loss_bc2(net, index):
#     x1, y1, x2, y2 = index2frame(index, grid_num)
#     x, y, condition = bc2(x1, y1, x2, y2)
#     output = net(torch.cat([x, y], dim=1))
#     return loss(output, condition)

# def loss_bc3(net, index):
#     x1, y1, x2, y2 = index2frame(index, grid_num)
#     x, y, condition = bc3(x1, y1, x2, y2)
#     output = net(torch.cat([x, y], dim=1))
#     return loss(output, condition)

# def loss_bc4(net, index):
#     x1, y1, x2, y2 = index2frame(index, grid_num)
#     x, y, condition = bc4(x1, y1, x2, y2)
#     output = net(torch.cat([x, y], dim=1))
#     return loss(output, condition)

# net = MLP().to(device)
net = torch.load('ordinary.pth')

def train():
    
    optimizer = torch.optim.Adam(params=net.parameters(), lr=1e-3)
    # optimizer = torch.optim.SGD(params=net.parameters(), momentum=0.5, lr=1e-3)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5e2, gamma=0.8)
    
    coef = 10
    quad_integral.init()
    test_func.init(5)
    for i in tqdm(range(10000)):
        # a = -216
        # b = 2000
        # coef = a * math.log(i + 1) + b
        optimizer.zero_grad()
        loss_tot = loss_interior_2(net) \
        + coef * (loss_bc1(net) + loss_bc2(net) + loss_bc3(net) + loss_bc4(net))
        # loss_tot = loss_interior_2(net) + coef * (loss_bc1(net) + loss_bc2(net) + loss_bc3(net) + loss_bc4(net))
        # print(loss_interior_1(net))
        # print(coef * (loss_bc1(net) + loss_bc2(net) + loss_bc3(net) + loss_bc4(net)))
            # loss_tot += loss_interior_1(net, index, 1, 1) + loss_interior_1(net, index, 1, 2) + loss_interior_1(net, index, 1, 3) + loss_interior_1(net, index, 1, 4) + loss_interior_1(net, index, 1, 5)\
            #         + loss_interior_1(net, index, 2, 1) + loss_interior_1(net, index, 2, 2) + loss_interior_1(net, index, 2, 3) + loss_interior_1(net, index, 2, 4) + loss_interior_1(net, index, 2, 5)\
            #         + loss_interior_1(net, index, 3, 1) + loss_interior_1(net, index, 3, 2) + loss_interior_1(net, index, 3, 3) + loss_interior_1(net, index, 3, 4) + loss_interior_1(net, index, 3, 5)\
            #         + loss_interior_1(net, index, 4, 1) + loss_interior_1(net, index, 4, 2) + loss_interior_1(net, index, 4, 3) + loss_interior_1(net, index, 4, 4) + loss_interior_1(net, index, 4, 5)\
            #         + loss_interior_1(net, index, 5, 1) + loss_interior_1(net, index, 5, 2) + loss_interior_1(net, index, 5, 3) + loss_interior_1(net, index, 5, 4) + loss_interior_1(net, index, 5, 5)\
    
        if i % 100 == 0:
            print(loss_tot)
        loss_tot.backward()
        optimizer.step()
    # print(loss_interior_1(net, 1), loss_interior_2(net, 2), loss_tot)
    
    
        # scheduler.step()
    
    # torch.save(net, 'Poisson2DBased.pth')
    torch.save(net, 'Poisson2D.pth')

    loss1_dict = []
    loss2_dict = []
    loss3_dict = []
    for j in range(test_num):
        loss1_dict.append({str(i+1): loss for i, loss in enumerate(loss1[j])})
        loss2_dict.append({str(i+1): loss for i, loss in enumerate(loss2[j])})
        loss3_dict.append({str(i+1): loss for i, loss in enumerate(loss3[j])})

    for j in range(test_num):
        with open(f"./json/Poisson2D/loss1_{j}.json", "w") as f:
            json.dump(loss1_dict[j], f)
    
        with open(f"./json/Poisson2D/loss2_{j}.json", "w") as f:
            json.dump(loss2_dict[j], f)
    
        with open(f"./json/Poisson2D/loss3_{j}.json", "w") as f:
            json.dump(loss3_dict[j], f)
            
if __name__ == '__main__':
    # cProfile.run('train()', 'result')
    # p = pstats.Stats('result')
    # p.strip_dirs().sort_stats(-1).print_stats()
    train()
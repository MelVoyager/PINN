import torch
from tqdm import tqdm
from net_class import MLP
from Lengendre import Lengendre
import json
import new_lengendre
import os, sys

os.chdir(sys.path[0])
pi = torch.pi
sin = torch.sin
cos = torch.cos
test_num = 3
N = 100

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = torch.device("mps") if torch.has_mps else "cpu"
print(f"Using {device} device")

lengendre = Lengendre()

def u(x, y):
    return (0.1 * torch.sin(2 * torch.pi * x) + torch.tanh(10 * x)) * torch.sin(2 * torch.pi * y)

def gradient_u_2order(x, y):
    u_xx = -0.4 * pi ** 2 * torch.sin(2 * torch.pi * x) * torch.sin(2 * torch.pi * y) + 100 * torch.pow(1 / torch.cosh(10 * x), 2) * torch.sin(2 * torch.pi * y)
    u_yy = -(0.1 * torch.sin(2 * torch.pi * x) + torch.tanh(10 * x)) * 4 * (pi ** 2) * torch.sin(2 * torch.pi * y)
    return torch.cat([u_xx, u_yy], dim=1)

def f(x, y, m=1, n=1, c=0.1, k=10):
    term1 = (4 * torch.pi**2 * m**2 * c * torch.sin(2 * torch.pi * m * x) +
             (2 * k**2 * torch.sinh(k * x) / torch.cosh(k * x)**3)) * torch.sin(2 * torch.pi * n * y)
    term2 = 4 * torch.pi**2 * n**2 * (c * torch.sin(2 * torch.pi * m * x) + torch.tanh(k * x)) * torch.sin(2 * torch.pi * n * y)
    return -(term1 + term2)

def rand_in_interval(size, l=-1, r=1):
    return (torch.rand(size) * (r - l) + torch.full(size, l)).to(device)

def interior(n=N * N):
    x = rand_in_interval((n, 1))
    y = rand_in_interval((n, 1))
    condition = f(x, y)
    return x.requires_grad_(True), y.requires_grad_(True), condition

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
    condition = f(x, y)
    return x.requires_grad_(True), y.requires_grad_(True), condition

def bc1(n=1000):
    x = rand_in_interval((n, 1), l=1, r=1)
    y = rand_in_interval((n, 1))
    condition = u(x, y)
    return x.requires_grad_(True), y.requires_grad_(True), condition

def bc2(n=1000):
    x = rand_in_interval((n, 1))
    y = rand_in_interval((n, 1), l=1)
    condition = u(x, y)
    return x.requires_grad_(True), y.requires_grad_(True), condition

def bc3(n=1000):
    x = rand_in_interval((n, 1), r=-1)
    y = rand_in_interval((n, 1))
    condition = u(x, y)
    return x.requires_grad_(True), y.requires_grad_(True), condition

def bc4(n=1000):
    x = rand_in_interval((n, 1))
    y = rand_in_interval((n, 1), r=-1)
    condition = u(x, y)
    return x.requires_grad_(True), y.requires_grad_(True), condition

def gradients(u, x, order=1):
    if order == 1:
        return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                   create_graph=True,
                                   only_inputs=True, )[0]
    else:
        return gradients(gradients(u, x), x, order=order - 1)

def integral(func, l=-1, r=1, density=N, multipier = None):
    if multipier==None :
        return torch.sum(func) * (r - l) / density
    else : 
        return torch.sum(torch.mul(func, multipier)) * (r - l) / density

loss = torch.nn.MSELoss()

loss1 = [[] for _ in range(test_num)]
loss2 = [[] for _ in range(test_num)]
loss3 = [[] for _ in range(test_num)]

def loss_interior_ordinary(net, k=1):
    x, y, condition = interior()
    output = net(torch.cat([x, y], dim=1))
    x_grad_2order = gradients(output, x, 2)
    y_grad_2order = gradients(output, y, 2)
    lhs = x_grad_2order + y_grad_2order
    loss1[k-1].append(loss(lhs, condition).item())
    # print(loss(lhs, condition))
    return loss(lhs, condition)

def loss_interior_1(net, k=1):
    x, y, condition = grid()
    output = net(torch.cat([x, y], dim=1))
    x_grad_2order = gradients(output, x, 2)
    y_grad_2order = gradients(output, y, 2)
    int1 = torch.sum(lengendre.v(x, k) * (x_grad_2order + y_grad_2order)) * ((2 / N) ** 2)
    int2 = torch.sum(lengendre.v(x, k) * condition) * ((2 / N) ** 2)
    int3 = torch.sum(lengendre.v(x, k) * gradient_u_2order(x, y)) * ((2 / N) ** 2)
    # int1 = integral(lengendre.v(x, k), multipier=(x_grad_2order + y_grad_2order))
    # int2 = integral(lengendre.v(x, k), multipier=condition)
    
    loss1[k-1].append(loss(int1, int2).item())
    # print(loss(int1, int2))
    return loss(int1, int2)

def gradient_u(x, y):
    dx = (0.2 * torch.pi * torch.cos(2*torch.pi*x) + 10 * (1 / torch.cosh(10*x))**2) * torch.sin(2 * torch.pi * y)
    dy = 2 * torch.pi * (0.1 * torch.sin(2*torch.pi*x) + torch.tanh(10*x)) * torch.cos(2*torch.pi*y)
    return torch.cat([dx, dy], dim=1)

def loss_interior_2(net, k=1):
    x, y, condition = grid()
    output = net(torch.cat([x, y], dim=1))
    x_grad_1order = gradients(output, x, 1)
    y_grad_1order = gradients(output, y, 1)
    net_gradient_u =  torch.cat([x_grad_1order, y_grad_1order], dim=1)
    
    new_lengendre.v(1, k, x, y)
    int1 = torch.abs(torch.sum(net_gradient_u * new_lengendre.v(1, k, x, y))) * ((2 / N) ** 2)
    int2 = torch.abs(torch.sum(condition * new_lengendre.v(0, k, x, y))) * ((2 / N) ** 2)
    int3 = torch.abs(torch.sum(gradient_u(x, y) * new_lengendre.v(1, k, x, y))) * ((2 / N) ** 2)
    # print(int1, int2, int3)
    if k - 1 < test_num:
        loss2[k-1].append(loss(int1, int2).item())
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
    x, y, condition = bc1()
    output = net(torch.cat([x, y], dim=1))
    return loss(output, condition)

def loss_bc2(net):
    x, y, condition = bc2()
    output = net(torch.cat([x, y], dim=1))
    return loss(output, condition)

def loss_bc3(net):
    x, y, condition = bc3()
    output = net(torch.cat([x, y], dim=1))
    return loss(output, condition)

def loss_bc4(net):
    x, y, condition = bc4()
    output = net(torch.cat([x, y], dim=1))
    return loss(output, condition)

net = MLP().to(device)
# net = torch.load('ordinary.pth')
optimizer = torch.optim.Adam(params=net.parameters())
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2e3, gamma=0.5)
coef = 10

for i in tqdm(range(1000)):
    optimizer.zero_grad()
    # loss_tot =  loss_interior_ordinary(net) + \
    loss_tot = loss_interior_1(net, 1) + loss_interior_1(net, 2) \
                + loss_interior_2(net, 1) + loss_interior_2(net, 2) + loss_interior_2(net, 3)\
                + loss_interior_2(net, 4) + loss_interior_2(net, 5) + loss_interior_2(net, 6)\
                + coef * (loss_bc1(net) + loss_bc2(net) + loss_bc3(net) + loss_bc4(net)) 
    # print(loss_interior_1(net, 1), loss_interior_2(net, 2), loss_tot)
    loss_tot.backward()
    optimizer.step()
    scheduler.step()
    
torch.save(net, 'Poisson2D.pth')

loss1_dict = []
loss2_dict = []
loss3_dict = []
for j in range(test_num):
    loss1_dict.append({str(i+1): loss for i, loss in enumerate(loss1[j])})
    loss2_dict.append({str(i+1): loss for i, loss in enumerate(loss2[j])})
    loss3_dict.append({str(i+1): loss for i, loss in enumerate(loss3[j])})

for j in range(test_num):
    with open(f"../json/Poisson2D/loss1_{j}.json", "w") as f:
        json.dump(loss1_dict[j], f)
    
    with open(f"../json/Poisson2D/loss2_{j}.json", "w") as f:
        json.dump(loss2_dict[j], f)
    
    with open(f"../json/Poisson2D/loss3_{j}.json", "w") as f:
        json.dump(loss3_dict[j], f)
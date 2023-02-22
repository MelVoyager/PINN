'''
### Helmholtz Problem
$$
u(x,y)=sin(\pi x)sin(4\pi y)
$$
$$
\Delta u+u=(1-17\pi^2)sin(\pi x)sin(4\pi y)
$$
'''

import torch
from tqdm import tqdm
from net_class import MLP
import copy

pi = torch.pi
sin = torch.sin

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = torch.device("mps") if torch.has_mps else "cpu"
print(f"Using {device} device")

def rand_in_interval(size, l=-1, r=1):
    return (torch.rand(size) * (r - l) + torch.full(size, l)).to(device)

def interior(n=5000):
    x = rand_in_interval((n, 1))
    y = rand_in_interval((n, 1))
    condition = (1 - 17 * (pi ** 2)) * sin(pi * x) * sin(4 * pi * y)
    return x.requires_grad_(True), y.requires_grad_(True), condition

def bc1(n=1000):
    x = rand_in_interval((n, 1))
    y = torch.full_like(x, -1)
    condition = torch.zeros_like(x)
    return x.requires_grad_(True), y.requires_grad_(True), condition

def bc2(n=1000):
    y = rand_in_interval((n, 1))
    x = torch.full_like(y, 1)
    condition = torch.zeros_like(y)
    return x.requires_grad_(True), y.requires_grad_(True), condition

def bc3(n=1000):
    x = rand_in_interval((n, 1))
    y = torch.full_like(x, 1)
    condition = torch.zeros_like(x)
    return x.requires_grad_(True), y.requires_grad_(True), condition

def bc4(n=1000):
    y = rand_in_interval((n, 1))
    x = torch.full_like(y, -1)
    condition = torch.zeros_like(y)
    return x.requires_grad_(True), y.requires_grad_(True), condition

def gradients(u, x, order=1):
    if order == 1:
        return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                   create_graph=True,
                                   only_inputs=True, )[0]
    else:
        return gradients(gradients(u, x), x, order=order - 1)

loss = torch.nn.MSELoss()
    
def loss_interior(net):
    x, y, condition = interior()
    output = net(torch.cat([x, y], dim=1))
    return loss(output + gradients(output, x, 2) + gradients(output, y, 2), condition)

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
optimizer = torch.optim.Adam(params=net.parameters())

def mean(X):
    return torch.mean(torch.tensor(X))

coef = 1
beta = 0.9

for i in tqdm(range(10000)):
    # if i % 10 == 0:
    loss_list = [loss_interior(net), loss_bc1(net), loss_bc2(net), loss_bc3(net), loss_bc4(net)]
    optimizer.zero_grad()
    grads = []
    for i in range(len(loss_list)):
        l = loss_list[i]
        l.backward()
        grads.append(copy.deepcopy([parms.grad for name, parms in net.named_parameters()]))
        optimizer.zero_grad()
            # loss_total += l
    
    max_res_loss = max([torch.max(
        torch.abs(grads[0][i])) for i in range(len(grads[0]))])
    mean_constrain_loss = []
    for i in range(1, len(loss_list)):
        mean_constrain_loss.append(
            mean([torch.mean(torch.abs(grads[i][j])) for j in range(len(grads[i]))]))
    coef = (max_res_loss / mean(mean_constrain_loss)) * (1 - beta) + coef * beta
    
    optimizer.zero_grad()
    loss_total = loss_interior(net)\
        + coef * (loss_bc1(net) + loss_bc2(net) + loss_bc3(net)+ loss_bc4(net))

    loss_total.backward()
    optimizer.step()
    print(coef)
    del loss_list, grads
    
torch.save(net, 'Helmholtz.pth')
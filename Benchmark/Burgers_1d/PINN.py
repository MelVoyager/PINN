import torch
from tqdm import tqdm
from net_class import MLP

pi = torch.pi
sin = torch.sin

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = torch.device("mps") if torch.has_mps else "cpu"
print(f"Using {device} device")

def rand_in_interval(size, l=-1, r=1):
    return (torch.rand(size) * (r - l) + torch.full(size, l)).to(device)

def interior(n=5000):
    x = rand_in_interval((n, 1))
    y = rand_in_interval((n, 1), l=0)
    condition = torch.zeros_like(x)
    return x.requires_grad_(True), y.requires_grad_(True), condition

def bc1(n=100):
    x = rand_in_interval((n, 1))
    y = torch.full_like(x, 0)
    condition = -torch.sin(torch.pi * x)
    return x.requires_grad_(True), y.requires_grad_(True), condition

def bc2(n=100):
    y = rand_in_interval((n, 1), l=0)
    x = torch.full_like(y, 1)
    condition = torch.zeros_like(y)
    return x.requires_grad_(True), y.requires_grad_(True), condition

# def bc3(n=1000):
#     x = rand_in_interval((n, 1))
#     y = torch.full_like(x, 1)
#     condition = torch.zeros_like(x)
#     return x.requires_grad_(True), y.requires_grad_(True), condition

def bc4(n=100):
    y = rand_in_interval((n, 1), l=0)
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
    u = net(torch.cat([x, y], dim=1))
    u_x = gradients(u, x, 1)
    u_xx = gradients(u, x, 2)
    u_y = gradients(u, y, 1)
    return loss(u_y + u * u_x - 0.01 / torch.pi * u_xx, condition)

def loss_bc1(net):
    x, y, condition = bc1()
    output = net(torch.cat([x, y], dim=1))
    return loss(output, condition)

def loss_bc2(net):
    x, y, condition = bc2()
    output = net(torch.cat([x, y], dim=1))
    return loss(output, condition)

# def loss_bc3(net):
#     x, y, condition = bc3()
#     output = net(torch.cat([x, y], dim=1))
#     return loss(output, condition)

def loss_bc4(net):
    x, y, condition = bc4()
    output = net(torch.cat([x, y], dim=1))
    return loss(output, condition)

net = MLP([2, 20, 20, 20, 1]).to(device)
optimizer = torch.optim.Adam(params=net.parameters())

def mean(X):
    return torch.mean(torch.tensor(X)).item()

coef = 1
beta = 0.9

for i in tqdm(range(10000)):
    optimizer.zero_grad()
    loss_tot = loss_interior(net) + 10 * (loss_bc1(net) + loss_bc2(net) + loss_bc4(net))
    loss_tot.backward()
    optimizer.step()
    # if i % 10 == 0:
    #     loss_res_weights = []
    #     loss_bcs_weights = []
    
    #     # 计算max_loss_res
    #     optimizer.zero_grad()
    #     loss_res = loss_interior(net)
    #     loss_res.backward()
    #     for name, para in net.named_parameters():
    #         if "weight" in name:
    #             loss_res_weights.append(torch.max(torch.abs(para.grad)))
    #     max_loss_res = max(loss_res_weights)
    
    #     # 计算mean_loss_bcs
    #     optimizer.zero_grad()
    #     loss_bcs = coef * (loss_bc1(net) + loss_bc2(net) + loss_bc4(net))
    #     loss_bcs.backward()
    #     for name, para in net.named_parameters():
    #         if "weight" in name:
    #             loss_bcs_weights.append(torch.mean(torch.abs(para.grad)))
    #     mean_loss_bcs = mean(loss_bcs_weights) 
    
    #     # 更新coef并，加和得到loss_total
    #     coef = (max_loss_res / mean_loss_bcs) * (1 - beta) + coef * beta
    
    # optimizer.zero_grad()
    # loss_total = loss_interior(net) + coef * (loss_bc1(net) + loss_bc2(net) + loss_bc4(net))
    if i % 100 == 0:
        print(loss_tot)
    # loss_total.backward()
    # optimizer.step()
    # del loss_list, grads
    
torch.save(net, 'model/Burgers_PINN.pth')
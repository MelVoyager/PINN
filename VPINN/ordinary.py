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
    condition = -(pi ** 2) / 4 * sin(pi * x / 2)
    return x.requires_grad_(True), condition

def bc1(n=1000):
    x = rand_in_interval((n, 1), r= -1)
    condition = torch.full_like(x, -1)
    return x.requires_grad_(True), condition

def bc2(n=1000):
    x = rand_in_interval((n, 1), l= 1)
    condition = torch.full_like(x, 1)
    return x.requires_grad_(True), condition

def gradients(u, x, order=1):
    if order == 1:
        return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                   create_graph=True,
                                   only_inputs=True, )[0]
    else:
        return gradients(gradients(u, x), x, order=order - 1)

loss = torch.nn.MSELoss()

def loss_interior(net):
    x, condition = interior()
    output = net(x)
    return loss(gradients(output, x, 2), condition)

def loss_bc1(net):
    x, condition = bc1()
    output = net(x)
    return loss(output, condition)

def loss_bc2(net):
    x, condition = bc2()
    output = net(x)
    return loss(output, condition)

net = MLP().to(device)
optimizer = torch.optim.Adam(params=net.parameters())

coef = 10

for i in tqdm(range(10000)):
    optimizer.zero_grad()
    loss_tot = loss_interior(net) + coef * (loss_bc1(net) + loss_bc2(net))
    loss_tot.backward()
    optimizer.step()
    
torch.save(net, 'model/ordinary.pth')

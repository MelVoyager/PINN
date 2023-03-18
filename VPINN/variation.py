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


def interior(n=10000):
    x = torch.linspace(-1, 1, n).reshape(-1, 1)
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

def integral(func, l=-1, r=1, density=10000, multipier = None):
    if multipier==None :
        return torch.sum(func) * (r - l) / density
    else : 
        return torch.sum(torch.mul(func, multipier)) * (r - l) / density

loss = torch.nn.MSELoss()
def v(x, k=1):
    return sin(k * pi * x)

def loss_interior_1(net):
    x, condition = interior()
    output = net(x)
    net_grad_2order = gradients(output, x, 2)
    v1 = v(x)
    int1 = integral(v1, multipier=net_grad_2order)
    int2 = integral(v1, multipier=condition)
    return loss(int1, int2)

def loss_interior_2(net):
    x, condition = interior()
    output = net(x)
    net_grad_1order = gradients(output, x, 1)
    v1 = v(x)
    v_grad_1order = gradients(v1, x, 1)
    int1 = integral(v_grad_1order, multipier=net_grad_1order)
    int2 = integral(v1, multipier=condition)
    return loss(int1, int2)

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
    loss_tot = loss_interior_1(net) + loss_interior_2(net) + coef * (loss_bc1(net) + loss_bc2(net))
    loss_tot.backward()
    optimizer.step()
    
torch.save(net, 'variation.pth')

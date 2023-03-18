import torch
from tqdm import tqdm
from net_class import MLP
import json

pi = torch.pi
sin = torch.sin
test_num = 3

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

loss1 = [[] for _ in range(test_num)]
loss2 = [[] for _ in range(test_num)]
loss3 = [[] for _ in range(test_num)]

def loss_interior_1(net, k=1):
    x, condition = interior()
    output = net(x)
    net_grad_2order = gradients(output, x, 2)
    
    int1 = integral(v(x, k), multipier=net_grad_2order)
    int2 = integral(v(x, k), multipier=condition)
    
    loss1[k-1].append(loss(int1, int2).item())
    return loss(int1, int2)

def loss_interior_2(net, k=1):
    x, condition = interior()
    output = net(x)
    net_grad_1order = gradients(output, x, 1)
    v_grad_1order = gradients(v(x, k), x, 1)
    
    int1 = -integral(v_grad_1order, multipier=net_grad_1order)
    int2 = integral(v(x, k), multipier=condition)
    
    loss2[k-1].append(loss(int1, int2).item())
    return loss(int1, int2)

def loss_interior_3(net, k=1):
    x, condition = interior()
    output = net(x)
    v_grad_2order = gradients(v(x, k), x, 2)
    
    int1 = integral(v_grad_2order, multipier=output) + 2 * pi
    int2 = integral(v(x, k), multipier=condition)
    
    loss3[k-1].append(loss(int1, int2).item())
    return loss(int1, int2)

def loss_bc1(net):
    x, condition = bc1()
    output = net(x)
    return loss(output, condition)

def loss_bc2(net):
    x, condition = bc2()
    output = net(x)
    return loss(output, condition)

# net = MLP().to(device)
net = torch.load('ordinary.pth')
optimizer = torch.optim.Adam(params=net.parameters())

coef = 10

for i in tqdm(range(10000)):
    optimizer.zero_grad()
    loss_tot = loss_interior_1(net, 1) + loss_interior_2(net, 1) + loss_interior_3(net, 1) \
                + loss_interior_1(net, 2) + loss_interior_2(net, 2) + loss_interior_3(net, 2) \
                + loss_interior_1(net, 3) + loss_interior_2(net, 3) + loss_interior_3(net, 3) \
                + coef * (loss_bc1(net) + loss_bc2(net))
    loss_tot.backward()
    optimizer.step()
    
torch.save(net, 'variation.pth')

loss1_dict = []
loss2_dict = []
loss3_dict = []
for j in range(test_num):
    loss1_dict.append({str(i+1): loss for i, loss in enumerate(loss1[j])})
    loss2_dict.append({str(i+1): loss for i, loss in enumerate(loss2[j])})
    loss3_dict.append({str(i+1): loss for i, loss in enumerate(loss3[j])})

for j in range(test_num):
    with open(f"json/loss1_{j}.json", "w") as f:
        json.dump(loss1_dict[j], f)
    
    with open(f"json/loss2_{j}.json", "w") as f:
        json.dump(loss2_dict[j], f)
    
    with open(f"json/loss3_{j}.json", "w") as f:
        json.dump(loss3_dict[j], f)
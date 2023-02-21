'''
Helmholtz Problem

u(x,y)=sin(\pix)sin(4\piy)

'''

import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

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

# Neural Network
class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)


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
for i in tqdm(range(10000)):
    optimizer.zero_grad()
    loss_total = loss_interior(net) + loss_bc1(net)+ loss_bc2(net) + loss_bc3(net) + loss_bc4(net)
    loss_total.backward()
    optimizer.step()
    # if(i % 1000 == 0):print(i)
    
xc = torch.linspace(-1, 1, 500)
xx, yy = torch.meshgrid(xc, xc)
xx = xx.reshape(-1, 1)
yy = yy.reshape(-1, 1)
xy = torch.cat([xx, yy], dim=1)
prediction = net(xy)
res = prediction - sin(pi * xx) * sin(4 * pi * yy)
prediction = torch.reshape(prediction, (500, 500))
res = torch.reshape(res, (500, 500))

fig, ax = plt.subplots(nrows=1, ncols=2)
axes = ax.flatten()
image1 = axes[0].imshow(prediction.detach().numpy())
fig.colorbar(image1, ax=axes[0])
image2 = axes[1].imshow(res.detach().numpy())
fig.colorbar(image2, ax=axes[1])
fig.tight_layout()
plt.show()
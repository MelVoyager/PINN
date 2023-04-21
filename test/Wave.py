import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from net_class import MLP
import sys, os

os.chdir(sys.path[0])
pi = torch.pi
sin = torch.sin

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = torch.device("mps") if torch.has_mps else "cpu"
print(f"Using {device} device")

# define the pde
def u(x, t):
    term1 = torch.sin(torch.pi*x)*torch.cos(2*torch.pi*t)
    term2 = 0.5*torch.sin(4*torch.pi*x)*torch.cos(8*torch.pi*t)
    result = term1 + term2
    return result

def rand_in_interval(size, l=-1, r=1):
    return (torch.rand(size) * (r - l) + torch.full(size, l)).to(device)

def interior(n=10000):
    x = rand_in_interval((n, 1), l=0)
    y = rand_in_interval((n, 1), l=0)
    condition = torch.zeros_like(x)
    return x.requires_grad_(True), y.requires_grad_(True), condition

def bc1(n=100):
    x = rand_in_interval((n, 1), l=0)
    y = torch.full_like(x, 0)
    condition = sin(pi * x) + 0.5 * sin(4 * pi * x)
    return x.requires_grad_(True), y.requires_grad_(True), condition

def bc2(n=100):
    y = rand_in_interval((n, 1), l=0)
    x = torch.full_like(y, 1)
    condition = torch.zeros_like(y)
    return x.requires_grad_(True), y.requires_grad_(True), condition

def bc3(n=1000):
    x = rand_in_interval((n, 1), l=0)
    y = torch.full_like(x, 0)
    condition = torch.zeros_like(x)
    return x.requires_grad_(True), y.requires_grad_(True), condition

def bc4(n=100):
    y = rand_in_interval((n, 1), l=0)
    x = torch.full_like(y, 0)
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
    return loss(gradients(output, y, 2) - 4 * gradients(output, x, 2), torch.zeros_like(x))

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
    return loss(gradients(output, y, 1), condition)

def loss_bc4(net):
    x, y, condition = bc4()
    output = net(torch.cat([x, y], dim=1))
    return loss(output, condition)

net = MLP([2, 20, 20, 20, 20, 1]).to(device)
# net = torch.load('Wave_std.pth')
optimizer = torch.optim.Adam(params=net.parameters())

def mean(X):
    return torch.mean(torch.tensor(X)).item()

coef = 1

for i in tqdm(range(10000)):
    
    optimizer.zero_grad()
    loss_total = loss_interior(net) + coef * (loss_bc1(net) + loss_bc2(net) + loss_bc3(net) + loss_bc4(net))
    if i % 100 == 0:
        print(f'loss_interior={loss_interior(net).item():.5g}, loss_bc={loss_bc1(net) + loss_bc2(net) + loss_bc4(net):.5g}, coef={coef}')
    loss_total.backward()
    optimizer.step()
    # del loss_list, grads
    
torch.save(net, 'Wave_std.pth')


net = net.to('cpu')
xc = torch.linspace(0, 1, 500)
xx, yy = torch.meshgrid(xc, xc, indexing='ij')
xx = xx.reshape(-1, 1).requires_grad_(True)
yy = yy.reshape(-1, 1).requires_grad_(True)
xy = torch.cat([xx, yy], dim=1)
prediction = net(xy)
res = prediction - u(xx, yy)
prediction = torch.reshape(prediction, (500, 500))
res = torch.reshape(res, (500, 500))
solution = u(xx, yy).reshape(500, 500)

prediction = prediction.transpose(0, 1)
res = res.transpose(0, 1)
solution = solution.transpose(0, 1)

pde_residual = (gradients(prediction, yy, 2) - 4 * gradients(prediction, xx, 2)).reshape(500, 500).transpose(0, 1)

fig, ax = plt.subplots(nrows=2, ncols=2)
fig.set_figwidth(10)
fig.set_figheight(10)
axes = ax.flatten()

image1 = axes[0].imshow(prediction.detach().numpy(), cmap='jet', origin='lower', extent=[0, 1, 0, 1])
axes[0].set_title('Prediction')
fig.colorbar(image1, ax=axes[0])

image2 = axes[1].imshow(solution.detach().numpy(), cmap='jet', origin='lower', extent=[0, 1, 0, 1])
axes[1].set_title('solution')
fig.colorbar(image2, ax=axes[1])

image3 = axes[2].imshow(res.detach().numpy(), cmap='jet', origin='lower', extent=[0, 1, 0, 1])
axes[2].set_title(f'Residual,median={torch.median(torch.abs(res)):.3g}')
fig.colorbar(image3, ax=axes[2])

image4 = axes[3].imshow(pde_residual.detach().numpy(), cmap='jet', origin='lower', extent=[0, 1, 0, 1])
axes[3].set_title(f'PDE_Residual,median={torch.median(torch.abs(pde_residual)):.3g}')
fig.colorbar(image4, ax=axes[3])

fig.tight_layout()
plt.savefig("prediction_and_residual.png")
print(f'relative error={(torch.norm(res) / torch.norm(u(xx, yy))).item() * 100:.2f}%')
plt.show()
import matplotlib.pyplot as plt
import torch
import os, sys
from VPINN2d import VPINN
import numpy as np
# import pyinstrument
from copy import deepcopy
from matplotlib.tri import Triangulation
os.chdir(sys.path[0])

# import the f(rhs) and boundary condition, u is optional for plot
from Function.Burgers_1d import f, bc


device = 'cuda'
# train the model
vpinn = VPINN([2, 20, 20, 20, 1], f, bc(100, device=device), type=0, Q=10, grid_num=32, test_fcn_num=6, 
            device=device, load=None)
# with pyinstrument.Profiler() as prof:
net = vpinn.train("burgers1d", epoch_num=100, coef=0.0001)
# net = vpinn.train(None, epoch_num=0, coef=10)

# print(prof.output_text(unicode=True, color=True))
# verify
data = np.loadtxt('burgers1d.dat', skiprows=8)

# get x、y、u of solution
x_ = data[:, 0]
x = []
t = []
u = []
for i in range(11):
    x.append(deepcopy(x_))
    t.append([i * 0.1 for _ in range(len(x_))])
    u.append(data[:, i + 1])

x = np.concatenate(x)
t = np.concatenate(t)
u = np.concatenate(u)
tri = Triangulation(x, t)

x_tensor = torch.from_numpy(x).reshape(-1, 1).type(torch.float).to(device)
y_tensor = torch.from_numpy(t).reshape(-1, 1).type(torch.float).to(device)
u_tensor = torch.from_numpy(u).reshape(-1, 1).type(torch.float).to(device)

verify_tensor = net(torch.cat([x_tensor, y_tensor], dim=1))
print(f'median error={torch.median(torch.abs(u_tensor - verify_tensor))}')
print(f'relative error={torch.norm(u_tensor - verify_tensor) / torch.norm(u_tensor) * 100:.2f}%')

# plot and verify
xc = torch.linspace(-1, 1, 500)
yc = torch.linspace(0, 1, 500)
xx, yy = torch.meshgrid(xc, yc, indexing='ij')
xx = xx.reshape(-1, 1).to(device)
yy = yy.reshape(-1, 1).to(device)
xy = torch.cat([xx, yy], dim=1)
prediction = net(xy).to('cpu')
xx = xx.to('cpu')
yy = yy.to('cpu')
prediction = prediction.reshape(500, 500)
prediction = prediction.transpose(0, 1)
# plt.figure(figsize=(10,4))
# plt.imshow(prediction.detach().numpy(), cmap='hot', origin='lower', extent=[-1, 1, 0, 1])
# plt.colorbar()
# plt.savefig('prediction')
# plt.show()

fig, ax = plt.subplots(nrows=1, ncols=2)
fig.set_figwidth(14)
fig.set_figheight(5)
axes = ax.flatten()

# image 1
image1 = axes[0].imshow(prediction.detach().numpy(), cmap='hot', origin='lower', extent=[-1, 1, 0, 1])
axes[0].set_title('Prediction')
fig.colorbar(image1, ax=axes[0])

# image 2
tri = Triangulation(x, t)
res = (u_tensor - verify_tensor).to('cpu').reshape(-1).detach().numpy()
image2 = axes[1].tripcolor(tri, res, cmap='hot', edgecolors='k')
axes[1].set_title('Residual')
fig.colorbar(image2, ax=axes[1])
fig.tight_layout()
plt.savefig("prediction_and_residual.png")
# print(f'relative error={(torch.norm(res) / torch.norm(u(xx, yy))).item() * 100:.2f}%')
plt.show()
import matplotlib.pyplot as plt
import torch
import os, sys
from VPINN2d import VPINN
import numpy as np
os.chdir(sys.path[0])

# import the f(rhs) and boundary condition, u is optional for plot
# from ... import ...


device = 'cpu'
# train the model
vpinn = VPINN([2, 15, 15, 15, 1], f, bc(80, device=device), type=0, Q=10, grid_num=6, test_fcn_num=5, 
            device=device, load=None)
# net = vpinn.train("Poisson_Boltzmann", epoch_num=10000, coef=1)
net = vpinn.train(None, epoch_num=0, coef=10)


# verify
data = np.loadtxt('poisson_boltzmann2d.dat', skiprows=9)

# get x、y、u of solution
x = data[:, 0]
y = data[:, 1]
u = data[:, 2]

x_tensor = torch.from_numpy(x).reshape(-1, 1).type(torch.float).to(device)
y_tensor = torch.from_numpy(y).reshape(-1, 1).type(torch.float).to(device)
u_tensor = torch.from_numpy(u).reshape(-1, 1).type(torch.float).to(device)

prediction = net(torch.cat([x_tensor, y_tensor], dim=1))
print(f'median error={torch.median(torch.abs(u_tensor - prediction))}')
print(f'relative error={torch.norm(u_tensor - prediction) / torch.norm(u_tensor) * 100:.2f}%')

# plot and verify
xc = torch.linspace(-1, 1, 500)
xx, yy = torch.meshgrid(xc, xc, indexing='ij')
xx = xx.reshape(-1, 1).to(device)
yy = yy.reshape(-1, 1).to(device)
xy = torch.cat([xx, yy], dim=1)
prediction = net(xy).to('cpu')
xx = xx.to('cpu')
yy = yy.to('cpu')

prediction = prediction.reshape(500, 500)
prediction = prediction.transpose(0, 1)
plt.imshow(prediction.detach().numpy(), cmap='hot', origin='lower')
plt.colorbar()
plt.savefig('prediction')
plt.show()

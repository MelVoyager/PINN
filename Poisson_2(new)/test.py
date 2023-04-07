import matplotlib.pyplot as plt
import torch
import os, sys
from VPINN2d import VPINN
import numpy as np
# import the f(rhs) and boundary condition, u is optional for plot
from Function.Poisson_Boltzmann_2d import f, bc, in_circle
os.chdir(sys.path[0])

# train the model
vpinn = VPINN([2, 10, 10, 10, 1], f, bc(80), type=0, Q=10, grid_num=4, test_fcn_num=5, 
            device='cpu', load=None)
net = vpinn.train("Poisson_Boltzmann", epoch_num=10000, coef=10)
# net = vpinn.train(None, epoch_num=0, coef=10)


# verify
data = np.loadtxt('poisson_boltzmann2d.dat', skiprows=9)

# 提取x、y、u
x = data[:, 0]
y = data[:, 1]
u = data[:, 2]

x_tensor = torch.from_numpy(x).reshape(-1, 1).type(torch.float)
y_tensor = torch.from_numpy(y).reshape(-1, 1).type(torch.float)
u_tensor = torch.from_numpy(u).reshape(-1, 1).type(torch.float)

prediction = net(torch.cat([x_tensor, y_tensor], dim=1))
print(f'mean error={torch.mean(torch.abs(u_tensor - prediction))}')
print(f'relative error={torch.norm(u_tensor - prediction) / torch.norm(u_tensor) * 100:.2f}%')

# plot and verify
xc = torch.linspace(-1, 1, 500)
xx, yy = torch.meshgrid(xc, xc, indexing='ij')
xx = xx.reshape(-1, 1)
yy = yy.reshape(-1, 1)
xy = torch.cat([xx, yy], dim=1)
prediction = net(xy)
for i in range(500 ** 2):
    if in_circle(xx[i],yy[i]):
        # xx[i] = 
        # yy[i] = None
        prediction[i] = -1

prediction = prediction.reshape(500, 500)
prediction = prediction.transpose(0, 1)
plt.imshow(prediction.detach().numpy(), cmap='hot', origin='lower')
plt.colorbar()
plt.savefig('prediction')
plt.show()

# res = prediction - u(xx, yy)
# prediction = torch.reshape(prediction, (500, 500))
# res = torch.reshape(res, (500, 500))
# fig, ax = plt.subplots(nrows=1, ncols=2)
# fig.set_figwidth(13)
# fig.set_figheight(5)
# axes = ax.flatten()
# image1 = axes[0].imshow(prediction.detach().numpy())
# axes[0].set_title('Prediction')
# fig.colorbar(image1, ax=axes[0])
# image2 = axes[1].imshow(res.detach().numpy())
# axes[1].set_title('Solution')
# fig.colorbar(image2, ax=axes[1])
# fig.tight_layout()
# plt.savefig("prediction_and_residual.png")
# print(f'relative error={(torch.norm(res) / torch.norm(u(xx, yy))).item() * 100:.2f}%')
# plt.show()
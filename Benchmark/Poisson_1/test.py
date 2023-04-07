import matplotlib.pyplot as plt
import torch
import os, sys
from VPINN2d import VPINN
# import the f(rhs) and boundary condition, u is optional for plot
from Function.Sine_1 import u, f, bc
os.chdir(sys.path[0])

# train the model
vpinn = VPINN([2, 10, 10, 10, 1], f, bc(80), type=0, Q=10, grid_num=4, test_fcn_num=5, 
            device='cpu', load=None)
net = vpinn.train("Sine", epoch_num=20000)


# plot and verify
xc = torch.linspace(-1, 1, 500)
xx, yy = torch.meshgrid(xc, xc, indexing='ij')
xx = xx.reshape(-1, 1)
yy = yy.reshape(-1, 1)
xy = torch.cat([xx, yy], dim=1)
prediction = net(xy)
res = prediction - u(xx, yy)
prediction = torch.reshape(prediction, (500, 500))
res = torch.reshape(res, (500, 500))
fig, ax = plt.subplots(nrows=1, ncols=2)
fig.set_figwidth(13)
fig.set_figheight(5)
axes = ax.flatten()
image1 = axes[0].imshow(prediction.detach().numpy())
axes[0].set_title('Prediction')
fig.colorbar(image1, ax=axes[0])
image2 = axes[1].imshow(res.detach().numpy())
axes[1].set_title('Residual')
fig.colorbar(image2, ax=axes[1])
fig.tight_layout()
plt.savefig("prediction_and_residual.png")
print(f'relative error={(torch.norm(res) / torch.norm(u(xx, yy))).item() * 100:.2f}%')
plt.show()
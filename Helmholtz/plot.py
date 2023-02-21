import torch
import matplotlib.pyplot as plt
from net_class import MLP

pi = torch.pi
sin = torch.sin

net = torch.load('Helmholtz.pth')
xc = torch.linspace(-1, 1, 500)
xx, yy = torch.meshgrid(xc, xc, indexing='ij')
xx = xx.reshape(-1, 1)
yy = yy.reshape(-1, 1)
xy = torch.cat([xx, yy], dim=1)
prediction = net(xy)
res = prediction - sin(pi * xx) * sin(4 * pi * yy)
prediction = torch.reshape(prediction, (500, 500))
res = torch.reshape(res, (500, 500))

fig, ax = plt.subplots(nrows=1, ncols=2)
fig.set_figwidth(13)
fig.set_figheight(5)
axes = ax.flatten()
image1 = axes[0].imshow(prediction.detach().numpy())
fig.colorbar(image1, ax=axes[0])
image2 = axes[1].imshow(res.detach().numpy())
fig.colorbar(image2, ax=axes[1])
fig.tight_layout()
plt.savefig("prediction_and_residual.png")
plt.show()
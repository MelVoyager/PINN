import torch
import basis
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x = torch.linspace(-1, 1, 500)
y = torch.linspace(-1, 1, 500)

xx, yy = torch.meshgrid(x, y, indexing='ij')
xx = xx.reshape(-1, 1)
yy = yy.reshape(-1, 1)

f = basis.f(xx, yy)
laplace_u = torch.sum(basis.gradient_u_2order(xx, yy), dim=1, keepdim=True)
err = basis.f(xx, yy) - torch.sum(basis.gradient_u_2order(xx, yy), dim=1, keepdim=True)
err = err.reshape(500, 500)
f = f.reshape(500, 500)
laplace_u = laplace_u.reshape(500, 500)

plt.imshow(err)
plt.colorbar()
plt.show()
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # 绘制 3D 曲面图
# ax.plot_surface(xx.reshape(500, 500), yy.reshape(500, 500), f, cmap='viridis')

# # 设置轴标签
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')

# # 显示图形
# plt.show()
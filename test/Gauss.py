import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 定义高斯函数
def gaussian(x, y, z, mu, sigma):
    return np.exp(-((x - mu[0]) ** 2 + (y - mu[1]) ** 2 + (z - mu[2]) ** 2) / (2 * sigma ** 2))

# 生成数据
x = np.linspace(-5, 5, 200)
y = np.linspace(-5, 5, 200)
z = np.linspace(-5, 5, 200)
x, y, z = np.meshgrid(x, y, z)

mu = [0, 0, 0]  # 均值
sigma = 2       # 标准差
data = gaussian(x, y, z, mu, sigma)

# 绘制热力图
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(x, y, z, c=data.flatten(), cmap="inferno", marker="o", s=1, alpha=0.5, edgecolors="face")

ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")

plt.show()

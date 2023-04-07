import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
import os, sys
import torch
os.chdir(sys.path[0])
# 读取.dat文件中的数据
data = np.loadtxt('poisson_boltzmann2d.dat', skiprows=9)

# 提取x、y、u
x = data[:, 0]
y = data[:, 1]
u = data[:, 2]

tri = Triangulation(x, y)

# 绘制热力图
plt.tripcolor(tri, u, cmap='hot', edgecolors='k')
plt.colorbar()
plt.show()
# 绘制热力图
# plt.pcolormesh(x, y, np.array(u).reshape(len(y), len(x)), cmap='hot', shading='auto')
# plt.colorbar()
# plt.show()
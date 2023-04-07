import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from copy import deepcopy
import os, sys
import torch
os.chdir(sys.path[0])
# 读取.dat文件中的数据
data = np.loadtxt('burgers1d.dat', skiprows=8)

# 提取x、y、u
x_ = data[:, 0]
x = []
t = []
u = []
for i in range(11):
    x.append(deepcopy(x_))
    t.append([i for _ in range(len(x_))])
    u.append(data[:, i + 1])

x = np.concatenate(x)
t = np.concatenate(t)
u = np.concatenate(u)
tri = Triangulation(x, t)

# 绘制热力图
plt.tripcolor(tri, u, cmap='hot', edgecolors='k')
plt.colorbar()
plt.savefig('solution.png')
plt.show()
# 绘制热力图
# plt.pcolormesh(x, y, np.array(u).reshape(len(y), len(x)), cmap='hot', shading='auto')
# plt.colorbar()
# plt.show()
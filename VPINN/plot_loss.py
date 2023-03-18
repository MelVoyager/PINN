import json
import matplotlib.pyplot as plt

losses = [[[] for _ in range(3)] for _ in range(3)]
# 从 JSON 文件中读取 loss 值

for i in range(3):
    for j in range(3):
        # with open(f"json/Sine/loss{i+1}_{j}.json", "r") as f:
        with open(f"json/Lengendre/loss{i+1}_{j}.json", "r") as f:
            losses[i][j] = json.load(f)

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))

# 绘制 loss 随着轮数的变化图
for i in range(3):
    for j in range(3):
        axes[i, j].plot(list(losses[i][j].keys()), list(losses[i][j].values()))
        axes[i, j].set_title(f'loss {i+1}_{j}')

# plt.savefig("sine training losses.png")
plt.savefig("lengendre training losses.png")
# plt.show()

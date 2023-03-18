import json
import matplotlib.pyplot as plt

# 从 JSON 文件中读取 loss 值
with open("json/loss1_1.json", "r") as f:
    losses = json.load(f)

# 绘制 loss 随着轮数的变化图
plt.plot(list(losses.keys()), list(losses.values()))
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()

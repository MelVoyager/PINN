import torch
import torch.nn as nn
import torch.optim as optim

# 定义模型
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer1 = nn.Linear(10, 100)
        self.layer2 = nn.Linear(100, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

# 初始化模型
model = MyModel()

# 检查是否有多个 GPU 可用
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # model = nn.DataParallel(model)

# 将模型放到 GPU 上
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# 生成数据
data = torch.randn(10, 10).to(device)
target = torch.randn(10, 1).to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# 训练
optimizer.zero_grad()
output = model(data)
loss = criterion(output, target)
loss.backward()
optimizer.step()

import net_class
import torch
from net_class import MLP
import matplotlib.pyplot as plt
from scipy import integrate

pi = torch.pi
sin = torch.sin

# net = torch.load('model/variation.pth')
# net = torch.load('model/ordinary.pth')
net = torch.load('model/Lengendre_variation.pth')

x_test = torch.linspace(-1, 1, 200).reshape(-1, 1)
pred = net(x_test)
plt.plot(x_test.detach().numpy(), pred.detach().numpy(), color='red',linewidth=1.0,linestyle='--')
res = pred - sin(pi / 2 * x_test)
print(torch.max(torch.abs(res)))
plt.show()

def f(x):
    pred = net(torch.tensor(x).reshape(-1, 1))
    return pred.item()

v, err = integrate.quad(f, torch.tensor(-1), torch.tensor(1))
print(v)
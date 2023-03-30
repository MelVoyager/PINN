import torch
from net_class import MLP
from tqdm import tqdm
import os, sys
import matplotlib.pyplot as plt

os.chdir(sys.path[0])

class Lengendre:
    def P_0(self, x):
        return 1
    
    def P_0_prime(self, x):
        return 0
    
    def P_0_2prime(self, x):
        return 0
    
    def P_1(self, x):
        return x
    
    def P_1_prime(self, x):
        return 1
    
    def P_1_2prime(self, x):
        return 0
    
    def P_2(self, x):
        return 0.5 * (3 * x ** 2 - 1)
    
    def P_2_prime(self, x):
        return 3 * x
    
    def P_2_2prime(self, x):
        return 3
    
    def P_3(self, x):
        return 0.5 * (5 * x ** 3 - 3 * x)
    
    def P_3_prime(self, x):
        return 0.5 * (15 * x ** 2 - 3)
    
    def P_3_2prime(self, x):
         return 15 * x
    
    def P_4(self, x):
        return 0.125 * (35 * x ** 4 - 30 * x ** 2 + 3)
    
    def P_4_prime(self, x):
        return 0.5 * (35 * x ** 3 - 15 * x)
    
    def P_4_2prime(self, x):
        return 0.5 * (105 * x ** 2 - 15)
    
    def v(self, x, k=1):
        if k==1 :
            return self.P_2(x) - self.P_0(x)
        
        if k==2 :
            return self.P_3(x) - self.P_1(x)
        
        if k==3 :
            return self.P_4(x) - self.P_2(x)
        
    def v_prime(self, x, k=1):
         if k==1 : 
             return self.P_2_prime(x) - self.P_0_prime(x)
         
         if k==2 :
             return self.P_3_prime(x) - self.P_1_prime(x)
         
         if k==3 :
             return self.P_4_prime(x) - self.P_2_prime(x)
         
    def v_2prime(self, x, k=1):
         if k==1 : 
             return self.P_2_2prime(x) - self.P_0_2prime(x)
         
         if k==2 :
             return self.P_3_2prime(x) - self.P_1_2prime(x)
         
         if k==3 :
             return self.P_4_2prime(x) - self.P_2_2prime(x)
         
lengendre = Lengendre()

def u(x, y):
    return (0.1 * torch.sin(2 * torch.pi * x) + torch.tanh(10 * x)) * torch.sin(2 * torch.pi * y)

def f(x, y, m=1, n=1, c=0.1, k=10):
    term1 = (4 * torch.pi**2 * m**2 * c * torch.sin(2 * torch.pi * m * x) +
             (2 * k**2 * torch.sinh(k * x) / torch.cosh(k * x)**3)) * torch.sin(2 * torch.pi * n * y)
    term2 = 4 * torch.pi**2 * n**2 * (c * torch.sin(2 * torch.pi * m * x) + torch.tanh(k * x)) * torch.sin(2 * torch.pi * n * y)
    return -(term1 + term2)

def gradient_u(x, y):
    dx = (0.2 * torch.pi * torch.cos(2*torch.pi*x) + 10 * (1 / torch.cosh(10*x))**2) * torch.sin(2 * torch.pi * y)
    dy = 2 * torch.pi * (0.1 * torch.sin(2*torch.pi*x) + torch.tanh(10*x)) * torch.cos(2*torch.pi*y)
    return torch.cat([dx, dy], dim=1)

def v(x, y, k=1):
    return lengendre.v(x, k) * lengendre.v(y, k)

def gradient_v(x, y, k=1):
    return torch.cat([lengendre.v_prime(x, k) * lengendre.v(y, k), lengendre.v(x, k) * lengendre.v_prime(y, k)], dim=1)

N = 100
eps = torch.rand(1).item() * 1e-3
x = torch.linspace(-1+eps, 1-eps, N)
y = torch.linspace(-1+eps, 1-eps, N)
xx, yy = torch.meshgrid(x, y, indexing='ij')
xx = (xx.reshape(-1, 1)).requires_grad_(True)
yy = (yy.reshape(-1, 1)).requires_grad_(True)

s1 = 0
s2 = 0
s3 = 0

# print(torch.sum([torch.dot(gradient_u(x, y),gradient_v(x, y)) for (x, y) in (xx, yy)]))
def gradients(u, x, order=1):
    if order == 1:
        return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                   create_graph=True,
                                   only_inputs=True, )[0]
    else:
        return gradients(gradients(u, x), x, order=order - 1)


net = torch.load('ordinary.pth')
u = net(torch.cat([xx, yy], dim=1))
dx = gradients(u, xx, 1)
dy = gradients(u, yy, 1)
du = torch.cat([dx, dy], dim=1)






# s1 = torch.sum(gradient_u(xx, yy) * torch.cat([gradients(v(xx, yy, 2), xx), gradients(v(xx, yy, 2), yy)], dim=1))
s1 = torch.sum(gradient_u(xx, yy) * gradient_v(xx, yy, 2))
s2 = torch.sum(f(xx, yy) * v(xx, yy, 2))
s3 = torch.sum(du * gradient_v(xx, yy, 2))
print(s1, s2, s3)

# x_frame = torch.linspace(-1, 1, 100)
# y = lengendre.v_prime(x_frame, 3)
# plt.imshow((torch.sum(du * gradient_v(xx, yy, 2), dim=1, keepdim=True)).reshape(N, N).detach().numpy())
# plt.imshow((gradients(u, xx, order=2) + gradients(u, yy, order=2)).reshape(N, N).detach().numpy())
plt.imshow((f(xx, yy)).reshape(N, N).detach().numpy())
plt.colorbar()
plt.show()

# # torch., 
# for i in tqdm(range(N ** 2)):
#     s1 += torch.dot(gradient_u(xx[i], yy[i]), gradient_v(xx[i], yy[i]))
#     s2 += f(xx[i], yy[i]) * v(xx[i], yy[i])
#     s3 += torch.dot(torch.tensor([dx[i], dy[i]]), gradient_v(xx[i], yy[i]))
    
# print(s1, s2, s3)


    
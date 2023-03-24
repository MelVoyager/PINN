import torch

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
    dx = 0.2 * torch.pi * torch.cos(2*torch.pi*x) + 10 * (1 / torch.cosh(10*x))**2
    dy = 2 * torch.pi * (0.1 * torch.sin(2*torch.pi*x) + torch.tanh(10*x)) * torch.cos(2*torch.pi*y)
    return torch.tensor([dx, dy])

def v(x, y):
    return lengendre.v(x, 1) * lengendre.v(y, 1)

def gradient_v(x, y):
    return torch.tensor([lengendre.v_prime(x, 1) * lengendre.v(y, 1), lengendre.v(x, 1) * lengendre.v_prime(y, 1)])

x = torch.linspace(-1, 1, 100)
y = torch.linspace(-1, 1, 100)
xx, yy = torch.meshgrid(x, y, indexing='ij')
xx = xx.reshape(-1, 1)
yy = yy.reshape(-1, 1)

s1 = 0
s2 = 0

# print(torch.sum([torch.dot(gradient_u(x, y),gradient_v(x, y)) for (x, y) in (xx, yy)]))

for i in range(10000):
    s1 += torch.dot(gradient_u(xx[i], yy[i]),gradient_v(xx[i], yy[i]))
    s2 += f(xx[i], yy[i]) * v(xx[i], yy[i])
    
print(s1, s2)


    
import torch
from tqdm import tqdm
from net_class import MLP
import json

pi = torch.pi
sin = torch.sin
cos = torch.cos
test_num = 3

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = torch.device("mps") if torch.has_mps else "cpu"
print(f"Using {device} device")

class Lengendre:
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
            return self.P_2(x) - self.P_1(x)
        
        if k==2 :
            return self.P_3(x) - self.P_2(x)
        
        if k==3 :
            return self.P_4(x) - self.P_3(x)
        
    def v_prime(self, x, k=1):
         if k==1 : 
             return self.P_2_prime(x) - self.P_1_prime(x)
         
         if k==2 :
             return self.P_3_prime(x) - self.P_2_prime(x)
         
         if k==3 :
             return self.P_4_prime(x) - self.P_3_prime(x)
         
    def v_2prime(self, x, k=1):
         if k==1 : 
             return self.P_2_2prime(x) - self.P_1_2prime(x)
         
         if k==2 :
             return self.P_3_2prime(x) - self.P_2_2prime(x)
         
         if k==3 :
             return self.P_4_2prime(x) - self.P_3_2prime(x)
         
lengendre = Lengendre()
    
def rand_in_interval(size, l=-1, r=1):
    return (torch.rand(size) * (r - l) + torch.full(size, l)).to(device)

def interior(n=10000):
    x = torch.linspace(-1, 1, n).reshape(-1, 1)
    condition = -(pi ** 2) / 4 * sin(pi * x / 2)
    return x.requires_grad_(True), condition

def bc1(n=1000):
    x = rand_in_interval((n, 1), r= -1)
    condition = torch.full_like(x, -1)
    return x.requires_grad_(True), condition

def bc2(n=1000):
    x = rand_in_interval((n, 1), l= 1)
    condition = torch.full_like(x, 1)
    return x.requires_grad_(True), condition

def gradients(u, x, order=1):
    if order == 1:
        return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                   create_graph=True,
                                   only_inputs=True, )[0]
    else:
        return gradients(gradients(u, x), x, order=order - 1)

def integral(func, l=-1, r=1, density=10000, multipier = None):
    if multipier==None :
        return torch.sum(func) * (r - l) / density
    else : 
        return torch.sum(torch.mul(func, multipier)) * (r - l) / density

loss = torch.nn.MSELoss()

loss1 = [[] for _ in range(test_num)]
loss2 = [[] for _ in range(test_num)]
loss3 = [[] for _ in range(test_num)]

def loss_interior_1(net, k=1):
    x, condition = interior()
    output = net(x)
    net_grad_2order = gradients(output, x, 2)
    
    int1 = integral(lengendre.v(x, k), multipier=net_grad_2order)
    int2 = integral(lengendre.v(x, k), multipier=condition)
    
    loss1[k-1].append(loss(int1, int2).item())
    return loss(int1, int2)

def loss_interior_2(net, k=1):
    x, condition = interior()
    output = net(x)
    net_grad_1order = gradients(output, x, 1)
    int1 = -integral(lengendre.v_prime(x, k), multipier=net_grad_1order)
    int2 = integral(lengendre.v(x, k), multipier=condition)
    
    loss2[k-1].append(loss(int1, int2).item())
    return loss(int1, int2)

def loss_interior_3(net, k=1):
    x, condition = interior()
    output = net(x)
    int1 = integral(lengendre.v_2prime(x, k), multipier=output) \
        - (net(torch.full((1, 1), 1.)) * lengendre.v_prime(1, k) - net(torch.full((1, 1), -1.)) * lengendre.v_prime(-1, k))
    int1 = torch.sum(int1)
    int2 = integral(lengendre.v(x, k), multipier=condition)
    
    loss3[k-1].append(loss(int1, int2).item())
    return loss(int1, int2)

def loss_bc1(net):
    x, condition = bc1()
    output = net(x)
    return loss(output, condition)

def loss_bc2(net):
    x, condition = bc2()
    output = net(x)
    return loss(output, condition)

net = MLP().to(device)
# net = torch.load('model/ordinary.pth')
optimizer = torch.optim.Adam(params=net.parameters())

coef = 50

for i in tqdm(range(10000)):
    optimizer.zero_grad()
    loss_tot = loss_interior_1(net, 1) + loss_interior_2(net, 1) + loss_interior_3(net, 1) \
                + loss_interior_1(net, 2) + loss_interior_2(net, 2) + loss_interior_3(net, 2) \
                + loss_interior_1(net, 3) + loss_interior_2(net, 3) + loss_interior_3(net, 3) \
                + coef * (loss_bc1(net) + loss_bc2(net))
    loss_tot.backward()
    optimizer.step()
    
torch.save(net, 'model/Lengendre_variation.pth')

loss1_dict = []
loss2_dict = []
loss3_dict = []
for j in range(test_num):
    loss1_dict.append({str(i+1): loss for i, loss in enumerate(loss1[j])})
    loss2_dict.append({str(i+1): loss for i, loss in enumerate(loss2[j])})
    loss3_dict.append({str(i+1): loss for i, loss in enumerate(loss3[j])})

for j in range(test_num):
    with open(f"json/Lengendre/loss1_{j}.json", "w") as f:
        json.dump(loss1_dict[j], f)
    
    with open(f"json/Lengendre/loss2_{j}.json", "w") as f:
        json.dump(loss2_dict[j], f)
    
    with open(f"json/Lengendre/loss3_{j}.json", "w") as f:
        json.dump(loss3_dict[j], f)
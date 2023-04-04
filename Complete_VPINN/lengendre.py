import torch 

def legendre(n, x):
    if n == 0:
        return torch.ones_like(x)
    elif n == 1:
        return x
    else:
        return ((2 * n - 1) * x * legendre(n - 1, x) - (n - 1) * legendre(n - 2, x)) / n

def legendre_derivative(n, x):
    # 处理 x = 1 或 -1 的特殊情况
    x_mask = (x == 1) | (x == -1)
    result = torch.zeros_like(x)
    
    # 计算勒让德多项式的导数
    if n == 0:
        return result
    elif n == 1:
        return torch.where(x_mask, torch.ones_like(x), torch.ones_like(x))
    else:
        result = (n * (legendre(n - 1, x) - x * legendre(n, x))) / (1 - x**2)
        result[x_mask] = 0.0
        return result

    
def u(k, n, x):
    if k == 0:
        return legendre(n, x)
    elif k == 1:
        return legendre_derivative(n, x)
    else:
        raise ValueError("k must be 0 or 1")

def v(k, n1, n2, x, y):
    if k == 0:
        return (legendre(n1 + 1, x) - legendre(n1 - 1, x)) * (legendre(n2 + 1, y) - legendre(n2 - 1, y))
    
    elif k == 1:
        return torch.cat([(legendre_derivative(n1 + 1, x) - legendre_derivative(n1 - 1, x)) * (legendre(n2 + 1, y) - legendre(n2 - 1, y)), (legendre(n1 + 1, x) - legendre(n1 - 1, x)) * (legendre_derivative(n2 + 1, y) - legendre_derivative(n2 - 1, y))], dim=1)
    
class Test_Func:
    
    def init(self, test_func_num):
        self.test_func_num = test_func_num
        
    def test_func(self, k, x, y):
        ret = 0
        
        for n1 in range(1, self.test_func_num + 1):
            for n2 in range(1, self.test_func_num + 1):
                ret = ret + v(k, n1, n2, x, y)
        return ret
        # return torch.sum(torch.tensor([[[v(k, n1, n2, x, y)] for n1 in range(self.test_func_num)] for n2 in range(self.test_func_num)]))
    
test_func = Test_Func()
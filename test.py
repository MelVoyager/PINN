import functools
import inspect
import re

class VPINN:
    @staticmethod
    def laplace(x, y, u):
        return x * y * u

    def __init__(self, func):
        source_code = inspect.getsource(func)
        laplace_term_pattern = r'\bVPINN.laplace\((.+)\)'
        laplace_term = re.search(laplace_term_pattern, source_code)
        calls_laplace = bool(laplace_term)

        if calls_laplace:
            self.pde1 = self.extract_laplace_term(func, laplace_term.group(1).strip())
            self.pde2 = self.replace_laplace_with_dummy(func)
            self.pde = None
        else:
            self.pde = func
            self.pde1 = None
            self.pde2 = None

    def Laplace(self, x_, y_, u_):
        return x_ * y_ * u_

    def extract_laplace_term(self, func, laplace_term):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_args = inspect.signature(func).bind(*args, **kwargs).arguments
            x, y, u = func_args["x"], func_args["y"], func_args["u"]
            return eval(laplace_term.replace('VPINN.laplace(x, y, u)', '3 * self.Laplace(x, y, u)'), globals(), {'x': x, 'y': y, 'u': u, 'self': self})
        return wrapper

    def replace_laplace_with_dummy(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            original_laplace = VPINN.laplace
            VPINN.laplace = lambda x, y, u: 0
            result = func(*args, **kwargs)
            VPINN.laplace = original_laplace
            return result
        return wrapper

def example_function(x, y, u):
    return x + y + u + 3 * VPINN.laplace(x, y, u)

obj = VPINN(example_function)
if obj.pde is not None:
    print(obj.pde(5, 3, 1))  # 输出: "Function not available"
else:
    print(obj.pde1(5, 3, 1))  # 输出: 45 (仅包含 Laplace 项)
    print(obj.pde2(5, 3, 1))  # 输出: 

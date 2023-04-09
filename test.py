import functools
import inspect
import re

class VPINN:
    @staticmethod
    def laplace(x):
        return x * 2

    def __init__(self, func):
        source_code = inspect.getsource(func)
        calls_laplace = bool(re.search(r'\bVPINN.laplace\b', source_code))

        if calls_laplace:
            # pde1仅含laplace项
            self.pde1 = self.replace_laplace_with_self_and_dummy(func, True)
            # pde2含除laplace之外的其他项
            self.pde2 = self.replace_laplace_with_self_and_dummy(func, False)
            self.pde = None
        else:
            self.pde = func
            self.pde1 = None
            self.pde2 = None

    def Laplace(self, x):
        return x * 2

    def replace_laplace_with_self_and_dummy(self, func, is_pde1):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            original_laplace = VPINN.laplace
            VPINN.laplace = self.Laplace
            result_with_self = func(*args, **kwargs)
            VPINN.laplace = lambda x: 0
            result_with_dummy = func(*args, **kwargs)
            VPINN.laplace = original_laplace

            return result_with_self - result_with_dummy if is_pde1 else result_with_dummy
        return wrapper

def example_function(x):
    return x + VPINN.laplace(x) ** 2

obj = VPINN(example_function)
if obj.pde is not None:
    print(obj.pde(5))  # 输出: "Function not available"
else:
    print(obj.pde1(5))  # 输出: 10 (仅包含 Laplace 项)
    print(obj.pde2(5))  # 输出: 5 (不包含 Laplace 项)

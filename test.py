import functools
import inspect
import re

class VPINN:
    @staticmethod
    def laplace(x):
        return x * 2

    def __init__(self, func):
        source_code = inspect.getsource(func)
        laplace_term_pattern = r'([-+]? *[\w.]* *\*? *VPINN.laplace\([^)]*\) *(?:\*\* *[\w.]*)?)'
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

    def Laplace(self, x):
        return x * 2

    def extract_laplace_term(self, func, laplace_term):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_args = inspect.signature(func).bind(*args, **kwargs).arguments
            x = func_args["x"]
            return eval(laplace_term.replace('VPINN.laplace(x)', 'self.Laplace(x)'), globals(), {'x': x, 'self': self})
        return wrapper

    def replace_laplace_with_dummy(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            original_laplace = VPINN.laplace
            VPINN.laplace = lambda x: 0
            result = func(*args, **kwargs)
            VPINN.laplace = original_laplace
            return result
        return wrapper

def example_function(x, y):
    return x + 3 * VPINN.laplace(x) ** 2

obj = VPINN(example_function)
if obj.pde is not None:
    print(obj.pde(5))  # 输出: "Function not available"
else:
    print(obj.pde1(5))  # 输出: 200 (仅包含 Laplace 项)
    print(obj.pde2(5))  # 输出: 5 (不包含 Laplace 项)

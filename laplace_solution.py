import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
solution = np.zeros((100, 100))

for x in range(100):
    for y in range(100):
        solution[x][y] = np.sin(0.01 * x) * np.sinh(0.01 * y)

# print(solution)
df = pd.DataFrame(solution)
df.to_csv("data/laplace_solution.csv")
plt.imshow(solution)
plt.colorbar()
plt.tight_layout()
# plt.show()
plt.savefig("fig/laplace_solution.png")
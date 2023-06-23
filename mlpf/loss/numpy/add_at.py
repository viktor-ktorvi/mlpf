import numpy as np

a = np.array([1, 2, 3, 4, 5, 6, 7])
source = np.array([0, 1, 1, 0, 1, 0, 0])
result: np.ndarray = np.zeros((2,))
np.add.at(result, source, a)
print(result)

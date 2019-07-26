import numpy as np
a = np.array([2,2,3,3,3,4,4,4,4,5,5,5,5,5])
b = np.array([1,2,2,3,3,3,4,4,4,4,5,5,5,5])
c = np.array([2,3,3,3,4,4,4,4,5,5,5,5,5,6])
d = b-c
d[d != 0] = -1
d += 1
print(d.shape, a.shape)
a = a * d
print(a)

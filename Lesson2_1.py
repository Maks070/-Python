import numpy as np
a = np.array([[1, 2, 3, 3, 1],
              [6, 8, 11, 10, 7]]).transpose()
print(a)


mean_a = a.mean(axis=0)

print(mean_a)


a_centered = np.subtract(a, mean_a)

print(a_centered)

a_centered_sp = a_centered.T[0] @ a_centered.T[1]

print(a_centered_sp)

print(a_centered_sp / (a_centered.shape[0] - 1))

print(np.cov(a.T)[0, 1])

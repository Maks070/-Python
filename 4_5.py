from functools import reduce


def my_f(x, y):
    return x * y


print(reduce(my_f, [el for el in range(100, 1001) if el % 2 == 0]))

my_list = [2, 2, 2, 7, 23, 1, 44, 44, 3, 2, 10, 7, 4, 11]


def my_f(l):
    return [el for el in l if l.count(el) == 1]


print(my_f(my_list))

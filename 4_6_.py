from itertools import cycle

my_list = [2, 2, 2, 7, 23, 1, 44, 44, 3, 2, 10, 7, 4, 11]


def my_f(l):
    c = 0
    for el in cycle(l):
        if c == len(l):
            break
        else:
            print(el)
            c += 1


print(my_f(my_list))

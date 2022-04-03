from itertools import count
from sys import argv


def my_f(num):
    for el in count(num):
        if el > num + 10:
            break
        else:
            print(el)


try:
    nsme, num = argv
    my_f(int(num))
except ValueError:
    print("Введены не корректные данные")

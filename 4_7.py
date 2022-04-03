from math import factorial

try:
    n = int(input('Введите целое число '))


    def fact(n):
        for i in range(1, n + 1):
            yield i


    for el in fact(n):
        print(f'factorial {el} = {factorial(el)}')
except ValueError:
    print('Введены не коректные данные')

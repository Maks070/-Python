def my_f(x, y):
    c = 1
    while y < 0:
        c *= x
        y += 1
    return 1 / c


x = float(input('Введите действительное положительное число: '))
y = int(input('Введите целое отрицательное число: '))
print(my_f(x, y))

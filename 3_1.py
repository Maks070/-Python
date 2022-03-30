def my_f(arg1, arg2):
    try:
        return arg1 / arg2
    except ZeroDivisionError:
        return 'На ноль делить нельзя'


print(my_f(int(input('Введите делимое число: ')), int(input('Введите делитель: '))))

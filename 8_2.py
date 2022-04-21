class Zero(Exception):
    def __str__(self):
        return f'На 0 делить нельзя'

class Val(Exception):
    def __str__(self):
        return 'Введены не коректные данные'
a = input('Введите делимое ')
b = input('Введите делитель ')


try:
    if a.isdigit() and b.isdigit():
        if int(b) == 0:
            raise Zero()
        else:
            print(int(a) // int(b))

    else:
        raise Val()
except Val as err:
    print(err)
except Zero as err:
    print(err)

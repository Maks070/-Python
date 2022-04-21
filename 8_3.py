class Digit(Exception):
    def __str__(self):
        return "Это не число"


my_list = []

while True:
    num = input('Введите число или Stop для завершения ')
    if num.lower() == 'stop':
        print(my_list)
        break
    try:
        if num.isdigit() == False:
            raise Digit()
        else:
            my_list.append(num)
    except Digit as err:
        print(err)
print('END')

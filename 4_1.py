from sys import argv

try:
    nsme, s_1, s_2, s_3 = argv

    print(int(s_1) * int(s_2) + int(s_3))
except ValueError:
    print('Введены не коректные данные')

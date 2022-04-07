with open('5_5.txt', 'x+', encoding='UTF-8') as my_file:
    my_file.write(input('Введите набор чисел разделённых пробелами: '))
    my_file.seek(0)
    try:
        print(sum(map(float, (my_file.read()).split())))
    except ValueError:
        print('Введены не коректные данные')

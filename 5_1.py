with open('5_1.txt', 'x', encoding="utf-8") as my_file:
    while True:
        stroka = input('Введите строку для записи в файл или '
                       'оставте пустую для завершения ')
        if stroka == "":
            break
        else:
            my_file.write(stroka + '\n')

print('END')

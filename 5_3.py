my_list = []
total = 0
with open('5_3.txt', 'r', encoding='utf-8') as my_file:
    print('Список сотрудников имеющих оклад ниже 20000:')
    for line in my_file:
        total += 1
        my_list.append(float(line.split()[1]))
        if float(line.split()[1]) < 20000:
            print(line.strip())

print(f'Колличество сотрудников: {total} человек. Средняя величина дохода на сотрудников равна {sum(my_list)/total}')
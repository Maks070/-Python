stroka = 0
slova = 0
with open('5_2.txt', 'r', encoding='utf-8') as my_file:
    for line in my_file:
        stroka += 1
        print(f'В строке {stroka} количество слов равно {len(line.split())}')
        slova += len(line.split())
print(f'В файле {my_file.name} {stroka} строк и {slova} слов')

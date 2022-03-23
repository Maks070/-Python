rev = int(input('Введите выручку фирмы: '))
exp = int(input('Введите издержки фирмы: '))
if exp > rev:
    print('Отрицательный финансовый результат')
elif exp == rev:
    print('Фирма не имеет прибыли')
else:
    profitability = int(f'{(((rev - exp) / rev) * 100):.0f}')
    print(f'Рентабельность фирмы {profitability} %')
    people = int(input('Введите количество сотрудников фирмы:'))
    profit = (rev - exp) / people
    print(f'Прибыль фирмы в расчёте на одного сотрудника равна {profit}')

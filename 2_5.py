my_list = [7, 5, 3, 3, 2]
num = int((input('Введите место в рейтинге: ')))

if num > max(my_list):
    my_list.insert(0, num)
elif num <= min(my_list):
    my_list.append(num)
else:
    for i in range(len(my_list)):
        if num > my_list[i]:
            my_list.insert(i, num)
            break
print(f"Пользователь ввёл число {num}. Результат: {','.join(map(str, my_list))}")

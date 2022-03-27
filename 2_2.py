num = int(input("сколько значений должен содержать список?: "))
my_list = []
for i in range(num):
    my_list.append(input('Введите что вы хотите добавить в список: '))
print(f'исходный список: {my_list}')
for i in range(0, len(my_list) if len(my_list) % 2 == 0 else len(my_list) - 1, 2):
    my_list[i], my_list[i + 1] = my_list[i + 1], my_list[i]
print(f'Изменённый список: {my_list}')

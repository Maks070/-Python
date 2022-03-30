def my_func(a, b, c):
    my_list = [a, b, c]
    return sum(my_list) - min(my_list)


print('Введите числа')
a = int(input())
b = int(input())
c = int(input())
print(my_func(a, b, c))

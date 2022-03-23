num = int(input('Введите целое положительное число: '))
total = 0
while num != 0:
    n = num % 10
    if n > total:
        total = n
    num = num // 10
print(total)

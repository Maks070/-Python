def my_f(x):
    num = 0
    for i in x:
        if i.isdigit():
            num += int(i)
    return num


total = 0
s = 0

while True:
    x = input('Введите строку чисел, разделённых пробелом или q  для выхода: ')
    s = my_f(x.split())
    if 'q' in x:
        total += s
        print(f'({s}) {total}')
        break
    else:
        total += s
    print(f'({s}) {total}')

print("END")

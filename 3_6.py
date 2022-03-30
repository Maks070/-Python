def my_f(x):
    my_list = x.split()
    final_list = []
    for el in my_list:
        if el.islower:
            total = 0
            for i in range(len(el)):
                if 97 <= ord(el[i]) <= 122:
                    total += 1
                    if total == len(el):
                        final_list.append(el.title())

    return final_list


x = input('Введит строку из слов, разделённых пробелом.')
s = (my_f(x))
print(*s)

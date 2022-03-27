stroka = input("Ведите строку изнескольких слов, разделённых пробелами: ")
stroka = stroka.split(" ")
for ind, i in enumerate(stroka, 1):
    print(f'{ind} {i[:10]}')

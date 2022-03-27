my_list = [2, 2.3, (5 + 6j), 'str', list('Список'), tuple('кортеж'), set('множство'),
           frozenset('не изменяемое множество'),
           dict(key='val'), True, (b'text'), None, ZeroDivisionError]
for i in range(len(my_list)):
    print(my_list[i], type(my_list[i]))

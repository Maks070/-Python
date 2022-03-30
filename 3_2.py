def my_f(name, surname, year, city, email, phone):
    return f'{name} {surname}. Возраст:{year}. Город проживания: {city}.' \
           f' Адрес элктронной почты: {email}. Номер телефона: {phone}'


print(my_f(name=input('Введит своё имя: '), surname=input('Введите фамилию: '), year=input("Укажите сколько вам лет: "),
           city=input("В каком городе проживаете: "), email=input("Адрес вашей электронной почты: "),
           phone=input("Номер телефона: ")))

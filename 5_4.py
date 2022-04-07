from textblob import TextBlob

with open('5_4.txt', 'r', encoding='utf-8') as my_file:
    with open('5_4_.txt', 'x', encoding='utf-8') as my_file_:
        for line in my_file:
            blob = TextBlob(line.strip())
            translate = blob.translate(to='ru')
            print(translate, file=my_file_)
print("Перевод окончен")

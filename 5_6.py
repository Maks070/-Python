my_dict = {}
with open('5_6.txt', 'r', encoding='utf-8') as my_file:
    for line in my_file:
        new = line.split()
        total = []
        for i in new:
            num = ""
            for el in i:
                if el.isdigit():
                    num += el
            if num != '':
                total.append(num)
        my_dict[line.split()[0]] = sum(map(float, total))
print(my_dict)

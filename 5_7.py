import json

my_dict = {}
average_profit = {}
with open('5_7.txt', 'r', encoding='utf-8') as my_file:
    for line in my_file:
        my_dict[line.split()[0]] = float(line.split()[2]) - float(line.split()[3])
num = sum([i for i in my_dict.values() if i > 0]) / len([i for i in my_dict.values() if i > 0])
average_profit['average_profit'] = num
my_list = [my_dict, average_profit]
with open('5_7.json', 'w', encoding='utf-8') as write_file:
    json.dump(my_list, write_file, ensure_ascii=False, indent=2)

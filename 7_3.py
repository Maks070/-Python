class Cell:
    def __init__(self, cell):
        self.cell = cell

    def __str__(self):
        return f'Количество клеток равно {self.cell}'

    def __add__(self, other):
        return self.cell + other.cell

    def __sub__(self, other):
        if self.cell < other.cell:
            return "Так не получиться. Первая клетка меньше второй"
        else:
            return self.cell - other.cell

    def __mul__(self, other):
        return self.cell * other.cell

    def __truediv__(self, other):
        return self.cell // other.cell

    def make_order(self):
        print(("😆" * 5 + '\n') * (self.cell // 5) + ("😆" * (self.cell % 5) + '\n'))


a = Cell(5)
b = Cell(0)
try:
    print(a / b)
except ZeroDivisionError as err:
    print('На ноль делить нельзя')

c = Cell(18)
c.make_order()

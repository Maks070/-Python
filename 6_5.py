class Stationery:
    def __init__(self, title):
        self.title = title

    def draw(self):
        print('Запуск отрисовки')


class Pen(Stationery):
    def __init__(self):
        self.title = 'Pen'

    def draw(self):
        print(f'Отрисовка {self.title}')


class Pencil(Stationery):
    def __init__(self):
        self.title = 'Pencil'

    def draw(self):
        print(f'Отрисовка {self.title}')


class Handle(Stationery):
    def __init__(self):
        self.title = 'Handle'

    def draw(self):
        print(f'Отрисовка {self.title}')


a = Pen()
a.draw()
b = Stationery(1)
b.draw()
c = Pencil()
c.draw()
d = Handle()
d.draw()

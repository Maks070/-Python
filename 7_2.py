from abc import ABC, abstractmethod


class Clothes:

    def __init__(self, cloth):
        self.t_cloth = cloth

    def __add__(self, other):
        return f'Количество всего потраченоой ткани равно: {self.t_cloth + other.t_cloth} '

    @abstractmethod
    def cloth(self):
        pass


class Coat(Clothes):
    def __init__(self, v, num=1):
        self.num = num
        self.v = v
        self.t_cloth = (self.v / 6.5 + 0.5) * self.num

    @property
    def num(self):
        return self.__num

    @num.setter
    def num(self, num):
        if num < 0:
            self.__num = abs(num)
        else:
            self.__num = num

    def cloth(self):
        return f'{self.t_cloth} ткани понадобиться для {self.num} пальто'


class Suit(Clothes):
    def __init__(self, h, num=1):
        self.num = num
        self.h = h
        self.t_cloth = (2 * self.h + 0.3) * self.num

    @property
    def num(self):
        return self.__num

    @num.setter
    def num(self, num):
        if num < 0:
            self.__num = abs(num)
        else:
            self.__num = num

    def cloth(self):
        return f'{self.t_cloth} ткани понадобиться для {self.num} костюма(ов)'


a = Coat(4, -2)
print(a.cloth())
b = Suit(3, -2)
print(b.cloth())
print(a + b)

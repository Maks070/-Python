from abc import abstractmethod


class Error(Exception):
    def __str__(self):
        return 'На складе столько нету'


class Sklad:
    __tehnika = {}

    def add_sklad(self, key, value):

        if self.__tehnika.get(key) == None:
            self.__tehnika[key] = 0
        self.__tehnika[key] += value

    def issued(self, key, val):
        qua = self.__tehnika.get(key)
        try:
            if qua == None or qua < val:
                raise Error()

            else:

                self.__tehnika[key] -= val
        except Error as err:
            print(err)

    @property
    def tehnika_sklad(self):
        for key, value in self.__tehnika.items():
            print(f"{key}: {value}")


class Tehnika:

    def __init__(self, model, price, quantity):
        self.model = model
        self.price = price
        self.quantity = quantity

    @abstractmethod
    def __str__(self):
        print('Tehnika')


class Printer(Tehnika):
    model = "Canon"
    price = 2000

    def __init__(self, quantity):
        self.quantity = quantity

    def __str__(self):
        return f"{self.model} {self.quantity} "


class Scaner(Tehnika):
    model = "Epson"
    price = 1000

    def __init__(self, quantity):
        self.quantity = quantity

    def __str__(self):
        return f"{self.model} {self.quantity} "


class Kseroks(Tehnika):
    model = "Kseroks"
    price = 5000

    def __init__(self, quantity):
        self.quantity = quantity

    def __str__(self):
        return f"{self.model} {self.quantity} "


a = Scaner(4)
print(a)
b = Printer(5)
print(b)
c = Kseroks(6)
print(c)
e = Sklad()
print(Scaner.model)
e.add_sklad(Scaner.model, 4)
e.add_sklad(Printer.model, 5)
e.add_sklad(Kseroks.model, 6)

e.tehnika_sklad

e.issued(Scaner.model, 10)

e.tehnika_sklad

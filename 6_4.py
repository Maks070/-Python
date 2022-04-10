from random import choice


class Car:
    def __init__(self, speed, color, name, is_police=False):
        self.speed = speed
        self.color = color
        self.name = name
        self.is_police = is_police
        self.direction = choice(['прямо', "направо", 'налево'])

    def go(self):
        print(f'{self.name} поехала ')

    def stop(self):
        print(f'{self.name} остановилась ')

    def turn(self):
        print(f'{self.name} похала {self.direction} ')

    def show_speed(self):
        print(f'Скорость {self.speed} км/ч')


class TownCar(Car):

    def show_speed(self):
        if self.speed > 60:
            print(f'\033[31mПревышние скорости на {self.speed - 60} км/ч')
        else:
            Car.show_speed(self)


class WorkCar(Car):
    def show_speed(self):
        if self.speed > 40:
            print(f'\033[31mПревышние скорости на {self.speed - 40} км/ч\033[0m')
        else:
            Car.show_speed(self)


class SportCar(Car):
    pass


class PoliceCar(Car):
    def __init__(self, speed, color, name, is_police=False):
        super().__init__(speed, color, name, is_police=True)


a = PoliceCar(60, 'Black', 'Lada')
print(f'is_police {a.is_police}')
a.go()
a.turn()
a.show_speed()
a.stop()

b = WorkCar(80, 'Black', 'Lada')
print(f'is_police {b.is_police}')
b.go()
b.turn()
b.show_speed()
b.stop()

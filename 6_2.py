class Road:

    def __init__(self, length, width):
        self._length = length
        self._widght = width
        self.weight = 25
        self.height = 5

    def asphalt_massa(self):
        asphalt_massa = self._length * self._widght * self.weight * self.height
        print(f'Масса асфальта необходимая для покрытия всей дороги равна {asphalt_massa/1000} т.')


a = Road(5000, 20)
a.asphalt_massa()

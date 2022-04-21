class Complex:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __add__(self, other):
        Complex.first = self.a + other.a
        Complex.second = self.b + other.b
        return f'({Complex.first}' \
               f'{"+" if Complex.second >= 0 else ""}' \
               f'{Complex.second}j)'

    def __mul__(self, other):
        Complex.first = self.a * other.a - self.b * other.b
        Complex.second = self.a * other.b + self.b * other.a
        return f'({Complex.first}' \
               f'{"+" if Complex.second >= 0 else ""}' \
               f'{Complex.second}j)'


a = Complex(2, 3)
b = Complex(1, -5)
print(a * b)
print(a + b)

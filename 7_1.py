class Matrix:
    def __init__(self, matrix):
        self.matrix = matrix

    def __str__(self):
        return "\n".join([" ".join(map(str, i)) for i in self.matrix])

    def __add__(self, other):
        try:
            return Matrix([[self.matrix[i][j] + other.matrix[i][j] for j in range
            (len(self.matrix[0]))] for i in range(len(self.matrix))])
        except IndexError:
            return 'Ошибка. Разноразмерные матрицы'


a = Matrix([[31, 22], [37, 43, 234], [2.3, 86, [1]], [1]])
b = Matrix([[31, 22], [37, 43], [51, 86, 123]])

print(a + b)
print(b)

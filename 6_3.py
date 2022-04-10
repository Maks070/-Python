class Worker:
    def __init__(self, name, surname, position, income):
        self.name = name
        self.surname = surname
        self.position = position
        self._income = income


class Position(Worker):

    def get_full_name(self):
        print(f'Имя сотрудника: {self.name} {self.surname}')

    def get_total_income(self):
        print(f"Доход сотрудника: {self._income['wage'] + self._income['bonus']}")


a = Position('Вася', 'Bdf', 'Должность', {'wage': 22, 'bonus': 23})
print(a.position)
a.get_full_name()
a.get_total_income()

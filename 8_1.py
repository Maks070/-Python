class Date:
    def __init__(self, d, m, y):
        self.d = d
        self.m = m
        self.y = y
        self.val(d, m, y)

    @classmethod
    def inter(cls, data):
        d, m, y = [int(i) for i in data.split(".")]
        return cls(d, m, y)

    @staticmethod
    def val(d, m, y):
        if 1 <= d <= 31 and 1 <= m <= 12 and 1940 <= y <= 2022:
            pass
        else:
            print('Не верно указана дата')


a = Date.inter('14.06.1990')
b = Date.inter('14.06.2044')
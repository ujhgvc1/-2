import copy
#Матрицы представлены как списки списков
def matrix_create(rows, cols, fill=0):
    #Создает матрицу заданного размер
    return [[fill for _ in range(cols)] for _ in range(rows)]

def matrix_transpose(matrix):
    #Транспонирование матрицы
    #zip(*matrix) распаковывает строки и собирает элементы по столбцам
    return [list(row) for row in zip(*matrix)]

def matrix_add(m1, m2):
    #Сложение двух матриw
    if len(m1) != len(m2) or len(m1[0]) != len(m2[0]):
        raise ValueError("Размерности матриц для сложения должны совпадать.")
    
    return [[m1[i][j] + m2[i][j] for j in range(len(m1[0]))] for i in range(len(m1))]

def matrix_scalar_mult(matrix, scalar):
    #Умножение матрицы на число
    return [[element * scalar for element in row] for row in matrix]

def matrix_mult(m1, m2):
    rows_a = len(m1)
    cols_a = len(m1[0])
    rows_b = len(m2)
    cols_b = len(m2[0])

    if cols_a != rows_b:
        raise ValueError(f"Нельзя умножить матрицу {rows_a}x{cols_a} на {rows_b}x{cols_b}")

    #создаем результирующую матрицу нулями
    result = matrix_create(rows_a, cols_b)

    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                result[i][j] += m1[i][k] * m2[k][j]
    return result

def matrix_minor(matrix, i, j):
    #вспомогательная функция: возвращает минор матрицы без строки i и столбца j
    return [row[:j] + row[j+1:] for row in (matrix[:i] + matrix[i+1:])]

def matrix_determinant(matrix):
    #вычисление определителя матрицы рекурсивно
    rows = len(matrix)
    cols = len(matrix[0])
    if rows != cols:
        raise ValueError("Определитель можно вычислить только для квадратной матрицы.")
    #Базовые случаи
    if rows == 1:
        return matrix[0][0]
    if rows == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    det = 0
    #Разложение по первой строке
    for col in range(cols):
        sign = (-1) ** col
        sub_det = matrix_determinant(matrix_minor(matrix, 0, col))
        det += sign * matrix[0][col] * sub_det
    return det

# Реализация через класс Matrix с перегрузкой операторов.
class Matrix:
    def __init__(self, data):
        #Инициализация матрицы.
        #data: список списков [[1, 2], [3, 4]]
        # Проверка на корректность структуры
        if not data or not isinstance(data, list) or not isinstance(data[0], list):
            raise ValueError("Данные должны быть списком списков")
        cols_len = len(data[0])
        for row in data:
            if len(row) != cols_len:
                raise ValueError("Все строки матрицы должны быть одной длины")   
        # Создаем глубокую копию, изменения внешней перем. не влияют на объект
        self.data = copy.deepcopy(data)
        self.rows = len(data)
        self.cols = cols_len

    def __str__(self):
        s = ""
        for row in self.data:
            s += "| " + " ".join(f"{val:4}" for val in row) + " |\n"
        return s

    def __repr__(self):
        return f"Matrix({self.data})"

    def __add__(self, other):
        #Перегрузка оператора + сложение матриц
        if not isinstance(other, Matrix):
            raise TypeError("Складывать можно только матрицы.")
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Размерности матриц не совпадают.")
        new_data = []
        for i in range(self.rows):
            new_row = []
            for j in range(self.cols):
                new_row.append(self.data[i][j] + other.data[i][j])
            new_data.append(new_row)
        return Matrix(new_data)

    def __mul__(self, other):
        #Перегрузка оператора * .
        #Поддерживает умножение Матрица * Скаляр и Матрица * Матрица.
        #Умножение на число
        if isinstance(other, (int, float)):
            new_data = [[self.data[i][j] * other for j in range(self.cols)] for i in range(self.rows)]
            return Matrix(new_data)
        #Умножение на другую матрицу
        elif isinstance(other, Matrix):
            if self.cols != other.rows:
                raise ValueError(f"Нельзя умножить матрицу {self.rows}x{self.cols} на {other.rows}x{other.cols}")
            result = [[0 for _ in range(other.cols)] for _ in range(self.rows)]
            for i in range(self.rows):
                for j in range(other.cols):
                    for k in range(self.cols):
                        result[i][j] += self.data[i][k] * other.data[k][j]
            return Matrix(result)
        else:
            raise TypeError("Умножение поддерживается только на число или другую матрицу.")

    def transpose(self):
        #возвращает новую транспонированную матрицу
        #используем zip(*self.data) для быстрой транспозиции
        new_data = [list(row) for row in zip(*self.data)]
        return Matrix(new_data)

    def determinant(self):
        #вычисление определителя. обертка над рекурсивным методом
        if self.rows != self.cols:
            raise ValueError("Определитель существует только для квадратных матриц.")
        return self._calc_determinant(self.data)

    def _calc_determinant(self, matrix_data):
        #внутренний рекурсивный метод для расчета определителя
        rows = len(matrix_data)
        if rows == 1:
            return matrix_data[0][0]
        if rows == 2:
            return matrix_data[0][0] * matrix_data[1][1] - matrix_data[0][1] * matrix_data[1][0]
        det = 0
        for col in range(len(matrix_data[0])):
            sign = (-1) ** col
            #получаем минор (удаляем 0-ю строку и текущий столбец)
            minor = [row[:col] + row[col+1:] for row in matrix_data[1:]]
            det += sign * matrix_data[0][col] * self._calc_determinant(minor)
        return det

if __name__ == "__main__":
    print("=== ТЕСТ ООП СТИЛЯ ===")
    
    #пример
    m1 = Matrix([[1, 2], [2, 3]])
    m2 = Matrix([[2, 5], [7, 9]])
    
    print(f"Матрица 1:\n{m1}")
    print(f"Матрица 2:\n{m2}")

    #сложение
    m3 = m1 + m2
    print(f"Сложение (m1 + m2):\n{m3}")

    #умножение матриц
    m4 = m1 * m2
    print(f"Умножение матриц (m1 * m2):\n{m4}")
    
    #умножение на число
    m_scalar = m1 * 10
    print(f"Умножение на скаляр (m1 * 10):\n{m_scalar}")

    #транспонирование
    m5 = m1.transpose()
    print(f"Транспонирование m1:\n{m5}")

    #определитель
    det = m1.determinant()
    print(f"Определитель m1: {det}")
    
    #тест определителя 3x3
    m_3x3 = Matrix([[6, 1, 1], [4, -2, 5], [2, 8, 7]])
    print(f"Матрица 3x3:\n{m_3x3}")
    print(f"Определитель 3x3: {m_3x3.determinant()}")


    print("\n" + "="*40 + "\n")
    print("ТЕСТ ФУНКЦИОНАЛЬНОГО СТИЛЯ")
    
    f_m1 = [[1, 2], [2, 3]]
    f_m2 = [[2, 5], [7, 9]]
    
    print(f"Списки 1: {f_m1}")
    print(f"Списки 2: {f_m2}")
    
    f_sum = matrix_add(f_m1, f_m2)
    print(f"Функция суммы: {f_sum}")
    
    f_mult = matrix_mult(f_m1, f_m2)
    print(f"Функция умножения матриц: {f_mult}")
    
    f_scalar = matrix_scalar_mult(f_m1, 10)
    print(f"Функция числового умножения: {f_scalar}")
    
    f_trans = matrix_transpose(f_m1)
    print(f"Функция транспонирования: {f_trans}")
    
    f_det = matrix_determinant(f_m1)
    print(f"Функция определителя: {f_det}")
def gaussian_elimination(coeff_matrix, rhs_vector):
    num_equations = len(rhs_vector)

    # Прямой ход: приведение матрицы к верхнетреугольному виду
    for pivot_row in range(num_equations):
        # Поиск максимального элемента в текущем столбце
        max_row = pivot_row
        for row in range(pivot_row + 1, num_equations):
            if abs(coeff_matrix[row][pivot_row]) > abs(coeff_matrix[max_row][pivot_row]):
                max_row = row

        # Меняем строки местами
        coeff_matrix[pivot_row], coeff_matrix[max_row] = coeff_matrix[max_row], coeff_matrix[pivot_row]
        rhs_vector[pivot_row], rhs_vector[max_row] = rhs_vector[max_row], rhs_vector[pivot_row]

        # Проверка на вырожденность
        if coeff_matrix[pivot_row][pivot_row] == 0:
            raise ValueError("Матрица вырожденная, решение невозможно.")

        # Преобразуем строки для получения нулей ниже главной диагонали
        for row in range(pivot_row + 1, num_equations):
            factor = coeff_matrix[row][pivot_row] / coeff_matrix[pivot_row][pivot_row]
            for col in range(pivot_row, num_equations):
                coeff_matrix[row][col] -= factor * coeff_matrix[pivot_row][col]
            rhs_vector[row] -= factor * rhs_vector[pivot_row]

    # Обратный ход: нахождение решений
    solution = [0.0] * num_equations
    for i in range(num_equations - 1, -1, -1):
        sum_terms = sum(coeff_matrix[i][j] * solution[j] for j in range(i + 1, num_equations))
        solution[i] = (rhs_vector[i] - sum_terms) / coeff_matrix[i][i]

    return solution




def vector_norm(vector, norm_type='euclidean'):

    if norm_type == 'euclidean':
        # Евклидова норма (L2-норма)
        return sum(x ** 2 for x in vector) ** 0.5
    elif norm_type == 'l1':
        # Норма-1 (L1-норма)
        return sum(abs(x) for x in vector)
    elif norm_type == 'linf':
        # Норма-бесконечность (Linf-норма)
        return max(abs(x) for x in vector)
    else:
        raise ValueError("Неизвестный тип нормы. Допустимые значения: 'euclidean', 'l1', 'linf'.")

def matrix_norm(matrix, norm_type='euclidean'):

    if norm_type == 'euclidean':
        # Евклидова норма (Frobenius norm)
        return sum(sum(x ** 2 for x in row) for row in matrix) ** 0.5
    elif norm_type == 'l1':
        # Норма-1 (максимальная сумма по столбцам)
        return max(sum(abs(row[i]) for row in matrix) for i in range(len(matrix[0])))
    elif norm_type == 'linf':
        # Норма-бесконечность (максимальная сумма по строкам)
        return max(sum(abs(x) for x in row) for row in matrix)
    else:
        raise ValueError("Неизвестный тип нормы. Допустимые значения: 'euclidean', 'l1', 'linf'.")

def dot_product(vector1, vector2):
    return sum(x * y for x, y in zip(vector1, vector2))

def matrix_vector_multiply(matrix, vector):
    return [dot_product(row, vector) for row in matrix]

def householder_reflection(matrix, norm_type='euclidean'):
    """
    Приведение матрицы к верхнетреугольному виду с использованием преобразований Хаусхолдера.

    Параметры:
    matrix : list of lists - матрица коэффициентов.
    norm_type : str - тип нормы для вычисления ('euclidean', 'l1', 'linf').

    Возвращает:
    upper_triangular : list of lists - верхнетреугольная матрица.
    orthogonal_transform : list of lists - ортогональная матрица преобразований.
    """
    n = len(matrix)
    upper_triangular = [row[:] for row in matrix]  # Копируем матрицу
    orthogonal_transform = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]  # Единичная матрица

    for column_index in range(n - 1):
        # Выбираем подматрицу для текущего столбца
        column_vector = [upper_triangular[row][column_index] for row in range(column_index, n)]
        unit_vector = [0.0] * len(column_vector)
        unit_vector[0] = vector_norm(column_vector, norm_type)  # Вектор, который будет использоваться для отражения

        # Вектор отражения
        reflection_vector = [
            column_vector[j] - unit_vector[j] if column_vector[0] >= 0 else column_vector[j] + unit_vector[j]
            for j in range(len(column_vector))
        ]
        reflection_vector_norm = vector_norm(reflection_vector, norm_type)

        # Если норма вектора отражения равна нулю, пропускаем преобразование
        if reflection_vector_norm == 0:
            continue

        reflection_vector = [x / reflection_vector_norm for x in reflection_vector]  # Нормализация

        # Матрица отражения Хаусхолдера
        reflection_matrix = [
            [1.0 if row == col else 0.0 for col in range(n)] for row in range(n)
        ]
        for row in range(column_index, n):
            for col in range(column_index, n):
                reflection_matrix[row][col] -= 2.0 * reflection_vector[row - column_index] * reflection_vector[col - column_index]

        # Применяем преобразование Хаусхолдера к верхнетреугольной матрице и ортогональной матрице
        upper_triangular = [
            [dot_product(reflection_matrix[row], [upper_triangular[col][k] for col in range(n)]) for k in range(n)]
            for row in range(n)
        ]
        orthogonal_transform = [
            [dot_product(orthogonal_transform[row], [reflection_matrix[col][k] for col in range(n)]) for k in range(n)]
            for row in range(n)
        ]

    return upper_triangular, orthogonal_transform


def solve_householder(matrix, vector, norm_type='euclidean'):
    n = len(matrix)
    # Применяем преобразование Хаусхолдера к матрице
    upper_triangle_matrix, orthogonal_matrix = householder_reflection(matrix, norm_type)

    # Преобразуем вектор правых частей
    transformed_vector = [dot_product(orthogonal_matrix[row], vector) for row in range(n)]

    # Решаем верхнетреугольную систему Rx = b_transformed
    solution = [0.0] * n
    for i in range(n - 1, -1, -1):
        solution[i] = (transformed_vector[i] - sum(upper_triangle_matrix[i][j] * solution[j] for j in range(i + 1, n))) / upper_triangle_matrix[i][i]

    return solution

def compute_residual(matrix, solution, rhs_vector):
    n = len(matrix)
    residual = [0.0] * n
    for i in range(n):
        residual[i] = dot_product(matrix[i], solution) - rhs_vector[i]
    return residual

def inverse_matrix_gauss_jordan(matrix):
    """
    Нахождение обратной матрицы методом Гаусса-Жордана.
    """
    n = len(matrix)
    augmented_matrix = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(matrix)]

    for pivot_row in range(n):
        # Поиск максимального элемента в текущем столбце
        max_row = pivot_row
        for row in range(pivot_row + 1, n):
            if abs(augmented_matrix[row][pivot_row]) > abs(augmented_matrix[max_row][pivot_row]):
                max_row = row

        # Меняем строки местами
        augmented_matrix[pivot_row], augmented_matrix[max_row] = augmented_matrix[max_row], augmented_matrix[pivot_row]

        # Проверка на вырожденность
        if augmented_matrix[pivot_row][pivot_row] == 0:
            raise ValueError("Матрица вырожденная, обратной матрицы не существует.")

        # Нормализация текущей строки
        pivot_element = augmented_matrix[pivot_row][pivot_row]
        for col in range(2 * n):
            augmented_matrix[pivot_row][col] /= pivot_element

        # Обнуление элементов выше и ниже диагонали
        for row in range(n):
            if row != pivot_row:
                factor = augmented_matrix[row][pivot_row]
                for col in range(2 * n):
                    augmented_matrix[row][col] -= factor * augmented_matrix[pivot_row][col]

    # Извлекаем обратную матрицу
    inverse_matrix = [row[n:] for row in augmented_matrix]
    return inverse_matrix


def compute_condition_number(matrix, norm_type='euclidean'):
    """
    Оценка числа обусловленности матрицы.
    """
    # Вычисляем норму матрицы
    matrix_norm_value = matrix_norm(matrix, norm_type)

    # Находим обратную матрицу
    inverse_matrix = inverse_matrix_gauss_jordan(matrix)

    # Вычисляем норму обратной матрицы
    inverse_norm_value = matrix_norm(inverse_matrix, norm_type)

    # Число обусловленности
    return matrix_norm_value * inverse_norm_value


def main():
    print("Введите коэффициенты расширенной матрицы 4x5 построчно, разделяя числа пробелами:")
    augmented_matrix = []
    for _ in range(4):
        row = input().replace('@', '').split()
        row = list(map(float, row))
        augmented_matrix.append(row[:5])  # Берем только первые 5 столбцов

    # Разделяем коэффициенты и правую часть
    coeff_matrix = [row[:-1] for row in augmented_matrix]
    rhs_vector = [row[-1] for row in augmented_matrix]

    # Решаем СЛАУ методом Гаусса
    solution_gauss = gaussian_elimination(coeff_matrix, rhs_vector)
    print("Решение системы методом Гаусса:")
    for i, x in enumerate(solution_gauss):
        print(f"x{i + 1} = {x:.15f}")

    # Решаем СЛАУ методом Хаусхолдера
    solution_householder = solve_householder(coeff_matrix, rhs_vector, norm_type='euclidean')
    print("Решение системы методом Хаусхолдера:")
    for i, x in enumerate(solution_householder):
        print(f"x{i + 1} = {x:.15f}")
    residual_gauss = compute_residual(coeff_matrix, solution_gauss, rhs_vector)
    residual_householder = compute_residual(coeff_matrix, solution_householder, rhs_vector)

    print("Невязка метода Гаусса (L1):", vector_norm(residual_gauss, 'l1'))
    print("Невязка метода Гаусса (L∞):", vector_norm(residual_gauss, 'linf'))
    print("Невязка метода Хаусхолдера (L1):", vector_norm(residual_householder, 'l1'))
    print("Невязка метода Хаусхолдера (L∞):", vector_norm(residual_householder, 'linf'))

    # Находим обратную матрицу
    try:
        inverse_matrix = inverse_matrix_gauss_jordan(coeff_matrix)
        print("Обратная матрица:")
        for row in inverse_matrix:
            print(row)

        # Проверяем A * A^{-1} = E
        identity_matrix = [[sum(coeff_matrix[i][k] * inverse_matrix[k][j] for k in range(4)) for j in range(4)] for i in
                           range(4)]
        print("Проверка A * A^{-1} = E:")
        for row in identity_matrix:
            print(row)
    except ValueError as e:
        print(f"Ошибка при вычислении обратной матрицы: {e}")

    # Оцениваем число обусловленности
    condition_number = compute_condition_number(coeff_matrix, 'euclidean')
    print("Число обусловленности матрицы (евклидова норма):", condition_number)

    # Анализ влияния числа обусловленности на точность
    if condition_number < 10:
        print("Матрица хорошо обусловлена. Решение устойчиво.")
    elif 10 <= condition_number < 1000:
        print("Матрица умеренно обусловлена. Возможны небольшие погрешности.")
    else:
        print("Матрица плохо обусловлена. Решение может быть неточным.")


if __name__ == "__main__":
    main()

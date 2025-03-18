import numpy as np

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
    # Вычисляем норму матрицы
    matrix_norm_value = matrix_norm(matrix, norm_type)

    # Находим обратную матрицу
    inverse_matrix = inverse_matrix_gauss_jordan(matrix)

    # Вычисляем норму обратной матрицы
    inverse_norm_value = matrix_norm(inverse_matrix, norm_type)

    # Число обусловленности
    return matrix_norm_value * inverse_norm_value

def compute_errors(true_solution, approx_solution):
    n = len(true_solution)
    absolute_errors = [abs(true_solution[i] - approx_solution[i]) for i in range(n)]
    relative_errors = [absolute_errors[i] / abs(true_solution[i]) if true_solution[i] != 0 else 0 for i in range(n)]

    l1_absolute_error = sum(absolute_errors)
    linf_absolute_error = max(absolute_errors)

    l1_relative_error = sum(relative_errors)
    linf_relative_error = max(relative_errors)

    return l1_absolute_error, linf_absolute_error, l1_relative_error, linf_relative_error

def run_through_algorithm(a, c, b, d):
    n = len(c)
    # Копируем входные данные, чтобы не изменять оригинальные массивы
    a = a.copy()
    c = c.copy()
    b = b.copy()
    d = d.copy()

    # Прямой ход метода прогонки
    for i in range(1, n):
        factor = a[i - 1] / c[i - 1]
        c[i] -= factor * b[i - 1]
        d[i] -= factor * d[i - 1]

    # Обратный ход метода прогонки
    x = [0.0] * n
    x[-1] = d[-1] / c[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (d[i] - b[i] * x[i + 1]) / c[i]

    return x

def input_tridiagonal_system():
    print("Введите данные для трехдиагональной системы:")
    print("Введите компоненты вектора a (поддиагональ, начиная со второго элемента):")
    a = list(map(float, input().split()))

    print("Введите компоненты вектора c (диагональ):")
    c = list(map(float, input().split()))

    print("Введите компоненты вектора b (наддиагональ, без последнего элемента):")
    b = list(map(float, input().split()))

    print("Введите компоненты вектора d (правые части):")
    d = list(map(float, input().split()))

    return a, c, b, d

def calculate_residual_tridiagonal(a, c, b, d, x):
    n = len(c)
    r = [0.0] * n

    # Первая строка
    r[0] = c[0] * x[0] + b[0] * x[1] - d[0]

    # Средние строки
    for i in range(1, n - 1):
        r[i] = a[i - 1] * x[i - 1] + c[i] * x[i] + b[i] * x[i + 1] - d[i]

    # Последняя строка
    r[-1] = a[-1] * x[-2] + c[-1] * x[-1] - d[-1]

    return r
def run_through(a, c, b, d):
    x_tridiagonal = run_through_algorithm(a, c, b, d)
    print("Решение трехдиагональной системы методом прогонки:", x_tridiagonal)

    # Вычисление невязки
    residual_tridiagonal = calculate_residual_tridiagonal(a, c, b, d, x_tridiagonal)
    print("Невязка для трехдиагональной системы:", residual_tridiagonal)

    # Вычисление норм невязки
    norm_types = ['l1', 'linf']
    for norm_type in norm_types:
        residual_norm = vector_norm(residual_tridiagonal, norm_type)
        print(f"Норма невязки ({norm_type}-норма): {residual_norm}")

def main():
    print("Введите коэффициенты расширенной матрицы 4x5 построчно, разделяя числа пробелами. После каждой строки добавьте точное решение через символ @:")
    augmented_matrix = []
    true_solution = []

    for _ in range(4):
        row = input().strip()
        if '@' in row:
            # Разделяем строку на часть матрицы и точное решение
            matrix_part, true_solution_part = row.split('@')
            matrix_part = matrix_part.split()
            true_solution.append(float(true_solution_part.strip()))
        else:
            print("Ошибка: в строке отсутствует символ @.")
            return

        # Преобразуем часть строки матрицы в числа
        row_data = list(map(float, matrix_part))
        augmented_matrix.append(row_data[:5])  # Берем только первые 5 столбцов

    # Разделяем коэффициенты и правую часть
    coeff_matrix = [row[:-1] for row in augmented_matrix]
    rhs_vector = [row[-1] for row in augmented_matrix]

    # Решаем СЛАУ методом Гаусса
    solution_gauss = gaussian_elimination(coeff_matrix, rhs_vector)
    print("Решение системы методом Гаусса:")
    for i, x in enumerate(solution_gauss):
        print(f"x{i + 1} = {x}")

    # Решаем СЛАУ методом Хаусхолдера
    solution_householder = solve_householder(coeff_matrix, rhs_vector, norm_type='euclidean')
    print("Решение системы методом Хаусхолдера:")
    print(" ".join(f"x{i + 1} = {x}" for i, x in enumerate(solution_householder)))

    # Вычисляем погрешности для метода Гаусса
    l1_abs_err_gauss, linf_abs_err_gauss, l1_rel_err_gauss, linf_rel_err_gauss = compute_errors(true_solution, solution_gauss)
    print("Погрешности метода Гаусса:")
    print(f"L1 абсолютная погрешность: {l1_abs_err_gauss}")
    print(f"L∞ абсолютная погрешность: {linf_abs_err_gauss}")
    print(f"L1 относительная погрешность: {l1_rel_err_gauss}")
    print(f"L∞ относительная погрешность: {linf_rel_err_gauss}")

    # Вычисляем погрешности для метода Хаусхолдера
    l1_abs_err_householder, linf_abs_err_householder, l1_rel_err_householder, linf_rel_err_householder = compute_errors(true_solution, solution_householder)
    print("Погрешности метода Хаусхолдера:")
    print(f"L1 абсолютная погрешность: {l1_abs_err_householder}")
    print(f"L∞ абсолютная погрешность: {linf_abs_err_householder}")
    print(f"L1 относительная погрешность: {l1_rel_err_householder}")
    print(f"L∞ относительная погрешность: {linf_rel_err_householder}")

    # Вычисляем невязки
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
        identity_matrix = [[sum(coeff_matrix[i][k] * inverse_matrix[k][j] for k in range(4)) for j in range(4)] for i in range(4)]
        print("Проверка A * A^{-1} = E:")
        for row in identity_matrix:
            print(row)

        atol = 1e-6  # Абсолютная погрешность
        rtol = 1e-6  # Относительная погрешность

        if np.allclose(identity_matrix, np.eye(len(coeff_matrix)), atol=atol, rtol=rtol):
            print("Матрица A * A⁻¹ близка к единичной.")
        else:
            print("Матрица A * A⁻¹ не является единичной.")

    except ValueError as e:
        print(f"Ошибка при вычислении обратной матрицы: {e}")

    # Оцениваем число обусловленности
    condition_number = compute_condition_number(coeff_matrix, 'euclidean')
    print("Число обусловленности матрицы (евклидова норма):", condition_number)

    # Оцениваем число обусловленности
    condition_number = compute_condition_number(coeff_matrix, 'linf')
    print("Число обусловленности матрицы (linf норма):", condition_number)

    # Анализ влияния числа обусловленности на точность
    if condition_number < 10:
        print("Матрица хорошо обусловлена. Решение устойчиво.")
    elif 10 <= condition_number < 1000:
        print("Матрица умеренно обусловлена. Возможны небольшие погрешности.")
    else:
        print("Матрица плохо обусловлена. Решение может быть неточным.")

    a, c, b, d = input_tridiagonal_system()
    run_through(a, c, b, d)

if __name__ == "__main__":
    main()
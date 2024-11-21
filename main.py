import numpy as np
import time
import matplotlib.pyplot as plt

# Завантаження даних з файлу
data = np.loadtxt('y3.txt').T

# Відомі параметри системи
c2, c4, m2, m3 = 0.3, 0.12, 28, 18
t0, T, deltaT = 0, 50, 0.2   # Початковий час (t0), кінцевий час (T), крок по часу (deltaT)
epsilon = 1e-5   # Точність для завершення ітерацій

# Початкове наближення
c1, c3, m1 = 0.1, 0.1, 9

# Функція для розрахунку чутливості матриці для моделі
def SensMatrix(b):
    # Означення зворотних значень для m1, m3, b2 (в залежності від параметрів b)
    m1_inv, m3_inv, b2_inv = 1 / m1, 1 / m3, 1 / b[2] if b[2] != 0 else 0
    return np.array([
        [0, 1, 0, 0, 0, 0],
        [- (b[1] + b[0]) * m1_inv, 0, b[1] * m1_inv, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [b[1] * b2_inv, 0, -(b[1] + c3) * b2_inv, 0, c3 * b2_inv, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, c3 * m3_inv, 0, -(c4 + c3) * m3_inv, 0]
    ])

# Функція для розрахунку похідних моделі щодо параметрів
def ModelDerivatives(y, b):
    #  матриці для обчислення похідних по кожному параметру (db0, db1, db2)
    db0 = np.zeros((6, 6))
    db1 = np.zeros((6, 6))
    db2 = np.zeros((6, 6))

    # Заповнення значень для кожної матриці
    db0[1, 0] = -1 / m1
    db1[1, 0] = -1 / m1
    db1[1, 2] = 1 / m1
    db2[3, 0] = -b[1] / (b[2] ** 2)
    db2[3, 2] = (b[1] + c3) / (b[2] ** 2)
    db2[3, 4] = -c3 / (b[2] ** 2)

    # Множення на y
    db0 = np.dot(db0, y)
    db1 = np.dot(db1, y)
    db2 = np.dot(db2, y)

    return np.array([db0, db1, db2]).T

# Функція для обчислення чутливості за допомогою методу Рунге-Кутта
def Sensitivity_RK(A, db, uu, deltaT, timeStamps):
    # Розв'язання чутливості по методу Рунге-Кутти
    for i in range(1, len(timeStamps)):
        k1 = deltaT * (np.dot(A, uu[i - 1]) + db[i - 1])
        k2 = deltaT * (np.dot(A, (uu[i - 1] + k1 / 2)) + db[i - 1])
        k3 = deltaT * (np.dot(A, (uu[i - 1] + k2 / 2)) + db[i - 1])
        k4 = deltaT * (np.dot(A, (uu[i - 1] + k3)) + db[i - 1])

        uu[i] = uu[i - 1] + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return uu

# Функція для моделювання за допомогою методу Рунге-Кутта
def Model_RK(b, timeStamps, deltaT):
    #ініціалізуємо результати
    yy = np.zeros_like(data)
    yy[0] = data[0].copy()
    A = SensMatrix(b)  # Обчислення матриці чутливості для заданих параметрів b

    # розв'язування рівнянь по методу Рунге-Куттf
    for i in range(1, len(timeStamps)):
        y_prev = yy[i - 1]
        k1 = deltaT * np.dot(A, y_prev)
        k2 = deltaT * np.dot(A, (y_prev + k1 / 2))
        k3 = deltaT * np.dot(A, (y_prev + k2 / 2))
        k4 = deltaT * np.dot(A, (y_prev + k3))
        yy[i] = y_prev + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return yy

# Функція для розрахунку зміщень параметрів (deltaB)
def DeltaB(uu, db, deltaT, timeStamps, data, b):
    # Різниця між реальними даними та модельним прогнозом
    diff_y = data - Model_RK(b, timeStamps, deltaT)

    # Визначення зміщень параметрів
    du = (np.array([u.T @ u for u in uu]) * deltaT).sum(0)
    du_inv = np.linalg.inv(du) # Обернена матриця
    uY = (np.array([uu[i].T @ diff_y[i] for i in range(len(timeStamps))]) * deltaT).sum(0)
    deltaB = du_inv @ uY # Обчислення зміщення

    return deltaB

# Функція для обчислення середньоквадратичної помилки (MSE)
def MSE(data, model):
    return np.mean((data - model) ** 2)

# Основна функція для пошуку параметрів
def Parameters(b, t0, T, deltaT, eps, max_iter=100):
    # Створення масиву часових міток
    timeStamps = np.linspace(t0, T, int((T - t0) / deltaT + 1))

    iteration_count = 0
    prev_b = b.copy()

    iteration_results = []  # Список для збереження результатів ітерацій
    iteration_times = []  # Зберігання часу виконання кожної ітерації

    # Основний цикл для пошуку параметрів
    while iteration_count < max_iter:
        iteration_count += 1

        start_iter_time = time.time()  # Початок часу для ітерації
        yy = Model_RK(b, timeStamps, deltaT)
        uu = np.zeros((len(timeStamps), 6, 3))
        db = ModelDerivatives(yy.T, b)
        A = SensMatrix(b)
        uu = Sensitivity_RK(A, db, uu, deltaT, timeStamps)
        deltaB = DeltaB(uu, db, deltaT, timeStamps, data, b)

        b += deltaB
        mse = MSE(data, Model_RK(b, timeStamps, deltaT))

        iteration_results.append((iteration_count, b.copy(), mse))

        end_iter_time = time.time()  # Кінець часу для ітерації
        iteration_times.append(end_iter_time - start_iter_time)

        if np.abs(deltaB).max() < eps:
            break

    return b, iteration_count, iteration_results, iteration_times


# Головна функція
if __name__ == "__main__":
    start_time = time.time()
    solution, iteration_count, iteration_results, iteration_times = Parameters(
        np.array([c1, c3, m1]), t0, T, deltaT, epsilon
    )
    end_time = time.time()
    execution_time = end_time - start_time

    solution = np.round(solution, 4)

    # Виведення результатів
    print("  Ітерація |                  Невідомі параметри                  |    Показник якості")
    print("-" * 90)

    for iteration, params, mse in iteration_results:
        print(f"{iteration:10} |     c1 = {params[0]:.6f}, c3 = {params[1]:.6f}, m1 = {params[2]:.6f}     |      {mse:.8f}")

    print("-" * 90)
    print(f"Знайдені параметри: c1 = {solution[0]:.6f}, c3 = {solution[1]:.6f}, m1 = {solution[2]:.6f}")
    print(
        f"Показник якості: Q = {MSE(data, Model_RK(solution, np.linspace(t0, T, int((T - t0) / deltaT + 1)), deltaT)):.8f}")
    print(f"Час виконання: {execution_time:.6f} секунд")

    # Графік : Зміни параметрів (c1, c3, m1) за ітераціями
    iterations = [iteration for iteration, _, _ in iteration_results]
    c1_values = [params[0] for _, params, _ in iteration_results]
    c3_values = [params[1] for _, params, _ in iteration_results]
    m1_values = [params[2] for _, params, _ in iteration_results]

    plt.figure(figsize=(10, 6))
    plt.plot(iterations, c1_values, label='c1')
    plt.plot(iterations, c3_values, label='c3')
    plt.plot(iterations, m1_values, label='m1')
    plt.xlabel('Ітерація')
    plt.ylabel('Значення параметра')
    plt.title('Зміни параметрів за ітераціями')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Графік : МSE за ітераціями
    mse_values = [mse for _, _, mse in iteration_results]

    plt.figure(figsize=(10, 6))
    plt.plot(iterations, mse_values, label='MSE', color='violet')
    plt.xlabel('Ітерація')
    plt.ylabel('MSE')
    plt.title('Показник якості за ітераціями')
    plt.grid(True)
    plt.show()

    # Графік : Зміна маси (m1) за ітераціями
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, m1_values, label='m1', color='green')
    plt.xlabel('Ітерація')
    plt.ylabel('m1')
    plt.title('Зміна маси m1 за ітераціями')
    plt.grid(True)
    plt.show()

    # Графік : Зміна жорсткості (c1, c3) за ітераціями
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, c1_values, label='c1', color='blue')
    plt.plot(iterations, c3_values, label='c3', color='orange')
    plt.xlabel('Ітерація')
    plt.ylabel('Значення жорсткості')
    plt.title('Зміна жорсткості за ітераціями')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Графік : Зміна часу виконання за ітераціями
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, iteration_times, label='Час виконання', color='red')
    plt.xlabel('Ітерація')
    plt.ylabel('Час (с)')
    plt.title('Час виконання за ітераціями')
    plt.grid(True)
    plt.show()

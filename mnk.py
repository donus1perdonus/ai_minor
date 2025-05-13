import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Исходные данные
x = np.array([10, 35, 60, 85, 110, 135])
y = np.array([11.2, 28.8, 43.2, 56.2, 67.8, 79.2])

# 1. Определение аппроксимирующих функций

# a. Линейная функция y = ax + b
def linear_func(x, a, b):
    return a * x + b

# b. Степенная функция y = βx^a
def power_func(x, beta, a):
    return beta * x**a

# c. Показательная функция y = βe^(ax)
def exp_func(x, beta, a):
    return beta * np.exp(a * x)

# d. Квадратичная функция y = ax^2 + bx + c
def quad_func(x, a, b, c):
    return a * x**2 + b * x + c

# 2. Подгонка параметров для каждой функции
popt_lin, pcov_lin = curve_fit(linear_func, x, y)
popt_pow, pcov_pow = curve_fit(power_func, x, y, p0=[1, 1])
popt_exp, pcov_exp = curve_fit(exp_func, x, y, p0=[1, 0.01])
popt_quad, pcov_quad = curve_fit(quad_func, x, y)

# 3. Вычисление суммы квадратов отклонений S(a,b)
def calculate_s(y_real, y_pred):
    return np.sum((y_pred - y_real)**2)

# Предсказанные значения для каждой функции
y_lin = linear_func(x, *popt_lin)
y_pow = power_func(x, *popt_pow)
y_exp = exp_func(x, *popt_exp)
y_quad = quad_func(x, *popt_quad)

# Вычисление S для каждой функции
s_lin = calculate_s(y, y_lin)
s_pow = calculate_s(y, y_pow)
s_exp = calculate_s(y, y_exp)
s_quad = calculate_s(y, y_quad)

# 4. Создание графиков
x_fit = np.linspace(min(x), max(x), 100)

# Отдельные графики в одном окне
plt.figure(figsize=(12, 8))

# Линейная функция
plt.subplot(2, 2, 1)
plt.plot(x_fit, linear_func(x_fit, *popt_lin), 'r-', label=f'Линейная: y={popt_lin[0]:.4f}x+{popt_lin[1]:.4f}\nS={s_lin:.2f}')
plt.scatter(x, y, color='blue')
plt.title('Линейная аппроксимация')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()

# Степенная функция
plt.subplot(2, 2, 2)
plt.plot(x_fit, power_func(x_fit, *popt_pow), 'g-', label=f'Степенная: y={popt_pow[0]:.4f}x^{popt_pow[1]:.4f}\nS={s_pow:.2f}')
plt.scatter(x, y, color='blue')
plt.title('Степенная аппроксимация')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()

# Показательная функция
plt.subplot(2, 2, 3)
plt.plot(x_fit, exp_func(x_fit, *popt_exp), 'b-', label=f'Показательная: y={popt_exp[0]:.4f}e^{popt_exp[1]:.4f}x\nS={s_exp:.2f}')
plt.scatter(x, y, color='blue')
plt.title('Показательная аппроксимация')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()

# Квадратичная функция
plt.subplot(2, 2, 4)
plt.plot(x_fit, quad_func(x_fit, *popt_quad), 'm-', label=f'Квадратичная: y={popt_quad[0]:.4f}x²+{popt_quad[1]:.4f}x+{popt_quad[2]:.4f}\nS={s_quad:.2f}')
plt.scatter(x, y, color='blue')
plt.title('Квадратичная аппроксимация')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

# График всех функций в одном окне
plt.figure(figsize=(10, 6))
plt.plot(x_fit, linear_func(x_fit, *popt_lin), 'r-', label=f'Линейная (S={s_lin:.2f})')
plt.plot(x_fit, power_func(x_fit, *popt_pow), 'g-', label=f'Степенная (S={s_pow:.2f})')
plt.plot(x_fit, exp_func(x_fit, *popt_exp), 'b-', label=f'Показательная (S={s_exp:.2f})')
plt.plot(x_fit, quad_func(x_fit, *popt_quad), 'm-', label=f'Квадратичная (S={s_quad:.2f})')
plt.scatter(x, y, color='blue', label='Экспериментальные точки')
plt.title('Сравнение аппроксимирующих функций')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.show()

# Вывод результатов
print("Результаты аппроксимации:")
print(f"1. Линейная функция: y = {popt_lin[0]:.6f}x + {popt_lin[1]:.6f}, S = {s_lin:.6f}")
print(f"2. Степенная функция: y = {popt_pow[0]:.6f}x^{popt_pow[1]:.6f}, S = {s_pow:.6f}")
print(f"3. Показательная функция: y = {popt_exp[0]:.6f}e^{popt_exp[1]:.6f}x, S = {s_exp:.6f}")
print(f"4. Квадратичная функция: y = {popt_quad[0]:.6f}x² + {popt_quad[1]:.6f}x + {popt_quad[2]:.6f}, S = {s_quad:.6f}")

# Определение лучшей функции
functions = {
    'Линейная': s_lin,
    'Степенная': s_pow,
    'Показательная': s_exp,
    'Квадратичная': s_quad
}
best_func = min(functions, key=functions.get)
print(f"\nЛучшая аппроксимирующая функция: {best_func} (S = {functions[best_func]:.6f})")
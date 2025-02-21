import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Генерация случайных данных
def generate_data(num_points=100):
    np.random.seed(0)
    X = np.random.rand(num_points, 2)  # 100 точек в 2D
    y = np.random.randint(0, 2, num_points)  # 0 или 1 для двух классов
    return X, y

# Функция для вычисления расстояния
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

# Реализация k-NN
class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):
        # Вычисляем расстояния до всех точек в обучающем наборе
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        # Сортируем расстояния и выбираем k ближайших соседей
        k_indices = np.argsort(distances)[:self.k]
        # Получаем классы ближайших соседей
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # Возвращаем наиболее частый класс
        most_common = np.bincount(k_nearest_labels).argmax()
        return most_common

# Визуализация данных и предсказаний
def plot_decision_boundary(knn, X, y):
    # Создаем сетку для визуализации
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
    # Предсказания для каждой точки сетки
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Визуализация
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', marker='o')
    plt.title("k-NN Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

# Основной код
if __name__ == "__main__":
    X, y = generate_data(100)  # Генерация данных
    knn = KNN(k=5)  # Создание экземпляра KNN с k=5
    knn.fit(X, y)  # Обучение модели
    plot_decision_boundary(knn, X, y)  # Визуализация
import numpy as np

def sigmoid(x):
    """Функция сигмоиды."""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """Производная функции сигмоиды."""
    return x * (1 - x)

def hopfield_energy(weights, activation):
    """Вычисление энергии Хопфилда."""
    return -np.sum(activation * np.dot(weights, activation)) / 2

def update_activation(weights, activation):
    """Обновление активации нейронов."""
    activation = sigmoid(np.dot(weights, activation))
    return activation

def hopfield_network(input_patterns, epochs=100):
    """Реализация сети Хопфилда."""
    # Создание матрицы весов
    weights = np.zeros((len(input_patterns[0]), len(input_patterns[0])))
    for pattern in input_patterns:
        weights += np.outer(pattern, pattern)
    weights /= len(input_patterns)
    
    # Нормализация весов
    weights -= weights.mean()
    weights /= weights.std()
    
    # Предварительная обработка входных паттернов
    patterns = []
    for pattern in input_patterns:
        patterns.append(sigmoid(pattern))
    
    # Цикл обучения
    for _ in range(epochs):
        for pattern in patterns:
            activation = np.copy(pattern)
            while not np.all(activation == update_activation(weights, activation)):
                activation = update_activation(weights, activation)
    
    # Восстановление повреждённых паттернов
    recovered_patterns = []
    for pattern in patterns:
        activation = np.copy(pattern)
        while not np.all(activation == update_activation(weights, activation)):
            activation = update_activation(weights, activation)
        recovered_patterns.append(activation)
    
    return recovered_patterns

# Пример использования сети Хопфилда
# Предположим, что у нас есть следующие повреждённые арабские цифры:
damaged_digits = [
    [0.9, 0.1, 0.1, 0.9],
    [0.1, 0.9, 0.9, 0.1],
    [0.8, 0.2, 0.2, 0.8]
]

# Запускаем сеть Хопфилда
recovered_digits = hopfield_network(damaged_digits)

# Выводим результаты
for digit, recovered_digit in zip(damaged_digits, recovered_digits):
    print(f"Повреждённая цифра: {digit}, Восстановленная цифра: {recovered_digit}")

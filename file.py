import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cdist

# Загружаем набор данных Iris
iris = load_iris()
X, y = iris.data, iris.target

# Разделяем данные на параметрическую (P) и обучающую (T) выборки для каждого класса
X_train_list, X_param_list, y_train_list, y_param_list = [], [], [], []

print("Разделение на выборки для каждого класса:")
for i in range(len(iris.target_names)):
    indices = np.where(y == i)[0]
    train_indices, param_indices = train_test_split(indices, test_size=0.7, random_state=42)
    X_train_list.append(X[train_indices])
    X_param_list.append(X[param_indices])
    y_train_list.append(y[train_indices])
    y_param_list.append(y[param_indices])
    print(f"Класс {iris.target_names[i]} - Обучающая выборка: {len(train_indices)}, Параметрическая выборка: {len(param_indices)}")

# Превращаем списки в одномерные массивы
X_train = np.vstack(X_train_list)
X_param = np.vstack(X_param_list)
y_train = np.concatenate(y_train_list)
y_param = np.concatenate(y_param_list)

# Создаём массивы для хранения близостей
class_similarities = np.zeros((len(X_train), len(iris.target_names)))

#print("\nСредние расстояния от объектов обучающей выборки до параметрических объектов каждого класса:")
for i in range(len(iris.target_names)):
    X_param_class = X_param[y_param == i]
    similarities = cdist(X_train, X_param_class, metric='euclidean')
    class_similarities[:, i] = np.mean(similarities, axis=1)
#    print(f"Класс {iris.target_names[i]}: {class_similarities[:, i]}")

def decision_rule(similarities):
    return np.argmin(similarities, axis=1)

# Распознавание
y_pred = decision_rule(class_similarities)

# Оценка качества
correct_predictions = np.sum(y_pred == y_train)
total_predictions = len(y_train)
accuracy = correct_predictions / total_predictions
print(f'\nТочность: {accuracy:.4f}')

# Матрица ошибок
conf_matrix = confusion_matrix(y_train, y_pred)
print("\nМатрица ошибок:")
print(conf_matrix)

# =============================================================================
# # Вывод подробной информации о результатах классификации
# print("\nПодробные результаты:")
# for i in range(len(iris.target_names)):
#     print(f"Класс {iris.target_names[i]}:")
#     class_indices = np.where(y_train == i)[0]
#     for index in class_indices:
#         print(f"  Объект {index}, истинный класс: {iris.target_names[y_train[index]]}, "
#               f"предсказанный класс: {iris.target_names[y_pred[index]]}, "
#               f"средние расстояния: {class_similarities[index]}")
# =============================================================================

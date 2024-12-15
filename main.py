import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1. Pobranie bazy MNIST
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist.data, mnist.target

# Konwersja etykiet na liczby całkowite
y = y.astype(np.int32)

# 2. Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Normalizacja danych (standaryzacja)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Klasyfikator k-NN
k = 3  # liczba najbliższych sąsiadów
knn = KNeighborsClassifier(n_neighbors=k)

# Trenowanie modelu
knn.fit(X_train, y_train)

# 5. Predykcja na zbiorze testowym
y_pred = knn.predict(X_test)

# 6. Ocena modelu
accuracy = accuracy_score(y_test, y_pred)
print(f'Dokładność modelu k-NN (k={k}): {accuracy:.4f}')

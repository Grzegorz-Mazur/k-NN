# instal numpy, pandas, scikit-learn, matplotlib
#28x28  regukcja wymiarów, pca lub svd


import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import time
import matplotlib.pyplot as plt

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


# Funkcja do mierzenia czasu działania

def measure_time(method_name, X_train, X_test, y_train, y_test):
    start_time = time.time()

    # Tworzenie i trenowanie modelu k-NN
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)

    # Predykcja
    y_pred = knn.predict(X_test)

    # Pomiar czasu
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Dokładność
    accuracy = accuracy_score(y_test, y_pred)

    print(f"{method_name} - Czas wykonania: {elapsed_time:.4f} s, Dokładność: {accuracy:.4f}")
    return elapsed_time, accuracy


# 4. Metoda bez redukcji wymiarów
print("Bez redukcji wymiarów:")
time_original, accuracy_original = measure_time("Bez redukcji", X_train, X_test, y_train, y_test)

# 5. Metoda PCA
print("Z redukcją wymiarów PCA:")
pca = PCA(n_components=50)  # Redukcja do 50 wymiarów
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

time_pca, accuracy_pca = measure_time("PCA", X_train_pca, X_test_pca, y_train, y_test)

# 6. Metoda SVD
print("Z redukcją wymiarów SVD:")
svd = TruncatedSVD(n_components=50)  # Redukcja do 50 wymiarów
X_train_svd = svd.fit_transform(X_train)
X_test_svd = svd.transform(X_test)

time_svd, accuracy_svd = measure_time("SVD", X_train_svd, X_test_svd, y_train, y_test)

# 7. Wizualizacja porównania wyników
methods = ["Bez redukcji", "PCA", "SVD"]
times = [time_original, time_pca, time_svd]
accuracies = [accuracy_original, accuracy_pca, accuracy_svd]

# Wykres czasu działania
plt.figure(figsize=(10, 5))
plt.bar(methods, times, color='skyblue')
plt.title("Porównanie czasu wykonania")
plt.ylabel("Czas (s)")
plt.xlabel("Metoda")
plt.show()

# Wykres dokładności
plt.figure(figsize=(10, 5))
plt.bar(methods, accuracies, color='lightgreen')
plt.title("Porównanie dokładności")
plt.ylabel("Dokładność")
plt.xlabel("Metoda")
plt.ylim(0, 1)
plt.show()

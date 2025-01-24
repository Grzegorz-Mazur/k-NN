# instal numpy, pandas, scikit-learn, matplotlib
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

# Funkcja do wyświetlania obrazów
def display_images(images, labels, predictions=None, num=10):
    plt.figure(figsize=(10, 5))
    for i in range(num):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i].reshape(28, 28), cmap='gray')
        title = f'True: {labels[i]}'
        if predictions is not None:
            title += f'\nPred: {predictions[i]}'
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

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

# 7. Macierz pomyłek
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(10))
disp.plot(cmap='viridis', xticks_rotation=45)
plt.title('Macierz pomyłek')
plt.show()

# Konwersja y_test i y_train do numpy.array
y_test = np.array(y_test)
y_train = np.array(y_train)

# 8. Wyświetlenie przykładowych obrazów z predykcjami
display_images(X_test[:10], y_test[:10], y_pred[:10])

# 9. Analiza błędnych predykcji
incorrect = np.where(y_test != y_pred)[0]
display_images(X_test[incorrect], y_test[incorrect], y_pred[incorrect], num=10)

# 10. Szczegółowy raport klasyfikacji
report = classification_report(y_test, y_pred, target_names=[str(i) for i in range(10)])
print('Raport klasyfikacji:\n')
print(report)

# 11. Analiza odległości w k-NN
# Oblicz odległości dla pierwszego przykładu testowego
distances, indices = knn.kneighbors([X_test[0]])
print("Odległości do najbliższych sąsiadów:", distances)
print("Indeksy najbliższych sąsiadów:", indices)

# Wyświetl pierwszy przykład i jego sąsiadów
display_images(X_train[indices[0]], y_train[indices[0]], num=3)

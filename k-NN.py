# =============================================================================
# 1. Import bibliotek
# =============================================================================
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt


# =============================================================================
# 2. Funkcja do wyświetlania obrazów
# =============================================================================
def display_images(images, labels, predictions=None, num=10):
    """
    Wyświetla 'num' obrazów 28x28 wraz z oryginalną etykietą
    (oraz opcjonalnie predykcją).
    """
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


# =============================================================================
# 3. Pobranie bazy danych MNIST
# =============================================================================
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist.data, mnist.target

# Konwersja etykiet do typu całkowitego
y = y.astype(np.int32)

# =============================================================================
# 4. Podział na zbiór treningowy i testowy
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,  # 20% na test
    random_state=42
)

# =============================================================================
# 5. Normalizacja danych
# =============================================================================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =============================================================================
# 6. Definiujemy i trenujemy klasyfikator k-NN (scikit-learn)
# =============================================================================
k = 3
knn_sklearn = KNeighborsClassifier(n_neighbors=k)
knn_sklearn.fit(X_train, y_train)

# =============================================================================
# 7. Predykcja na zbiorze testowym (scikit-learn)
# =============================================================================
y_pred_sklearn = knn_sklearn.predict(X_test)

# =============================================================================
# 8. Ocena modelu (scikit-learn)
# =============================================================================
accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)
print(f"[scikit-learn] Dokładność modelu k-NN (k={k}): {accuracy_sklearn:.4f}")

cm_sklearn = confusion_matrix(y_test, y_pred_sklearn)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_sklearn, display_labels=np.arange(10))
disp.plot(cmap='viridis', xticks_rotation=45)
plt.title('Macierz pomyłek (scikit-learn)')
plt.show()

# =============================================================================
# 9. KONWERSJA y_test i y_train DO TABLIC NumPy
#    (aby nie wystąpił problem z indeksami w Pandas Series)
# =============================================================================
y_test = np.array(y_test)
y_train = np.array(y_train)

# Wyświetlenie przykładowych obrazów z predykcjami (scikit-learn)
display_images(X_test[:10], y_test[:10], y_pred_sklearn[:10])

# Analiza błędnych predykcji (scikit-learn)
incorrect_sklearn = np.where(y_test != y_pred_sklearn)[0]
display_images(X_test[incorrect_sklearn], y_test[incorrect_sklearn], y_pred_sklearn[incorrect_sklearn], num=10)

# Szczegółowy raport klasyfikacji (scikit-learn)
report_sklearn = classification_report(y_test, y_pred_sklearn, target_names=[str(i) for i in range(10)])
print('[scikit-learn] Raport klasyfikacji:\n')
print(report_sklearn)

# =============================================================================
# 10. Analiza odległości w k-NN scikit-learn (przykład dla pierwszej próbki)
# =============================================================================
distances, indices = knn_sklearn.kneighbors([X_test[0]])
print("Odległości do najbliższych sąsiadów:", distances)
print("Indeksy najbliższych sąsiadów:", indices)

# Wyświetlenie pierwszej próbki testowej i jej sąsiadów
display_images(X_train[indices[0]], y_train[indices[0]], num=3)


# =============================================================================
# 11. Własna implementacja k-NN
# =============================================================================
def my_knn_predict(X_train, y_train, X_test, k=3):
    """
    Moja własna (naiwna) implementacja algorytmu k-NN.

    Argumenty:
        X_train (ndarray): Dane treningowe, kształt (n_train, n_features).
        y_train (ndarray): Etykiety (klasy) dla danych treningowych, kształt (n_train,).
        X_test (ndarray): Dane testowe, kształt (n_test, n_features).
        k (int): Liczba najbliższych sąsiadów.

    Zwraca:
        ndarray: Predykcje dla zbioru X_test, kształt (n_test,).
    """
    predictions = np.zeros(X_test.shape[0], dtype=int)

    for i in range(X_test.shape[0]):
        # Oblicz odległości euklidesowe między X_test[i] a wszystkimi X_train
        distances = np.sqrt(np.sum((X_train - X_test[i]) ** 2, axis=1))

        # Pobierz indeksy k najmniejszych odległości
        k_smallest_indices = np.argsort(distances)[:k]

        # Znajdź klasy tych k najbliższych sąsiadów
        k_nearest_labels = y_train[k_smallest_indices]

        # Wybór najczęściej pojawiającej się klasy (majority vote)
        predicted_label = np.argmax(np.bincount(k_nearest_labels))

        predictions[i] = predicted_label

        # Opcjonalnie można dodać info o postępie:
        if i % 100 == 0 and i > 0:
             print(f"Przetworzono {i} / {X_test.shape[0]} próbek testowych...")

    return predictions


# =============================================================================
# 12. Porównanie wyników scikit-learn vs. własna implementacja
#     (ze względu na wydajność, zastosujemy podzbiór testowy w my_knn_predict)
# =============================================================================
subset_size = 2000  # Liczba próbek testowych do użycia w custom k-NN
X_test_subset = X_test[:subset_size]
y_test_subset = y_test[:subset_size]

print("\nRozpoczynam predykcję przy użyciu własnej implementacji k-NN (może to chwilę potrwać)...")
y_pred_custom = my_knn_predict(X_train, y_train, X_test_subset, k=3)

accuracy_custom = accuracy_score(y_test_subset, y_pred_custom)
print(f"[Custom k-NN] Dokładność na podzbiorze {subset_size} próbek: {accuracy_custom:.4f}")

cm_custom = confusion_matrix(y_test_subset, y_pred_custom)
disp_custom = ConfusionMatrixDisplay(confusion_matrix=cm_custom, display_labels=np.arange(10))
disp_custom.plot(cmap='viridis', xticks_rotation=45)
plt.title('Macierz pomyłek (Custom k-NN, subset testowy)')
plt.show()

display_images(X_test_subset[:10], y_test_subset[:10], y_pred_custom[:10])

report_custom = classification_report(y_test_subset, y_pred_custom, target_names=[str(i) for i in range(10)])
print('[Custom k-NN] Raport klasyfikacji (subset testowy):\n')
print(report_custom)

# =============================================================================
# 13. Podsumowanie
# =============================================================================
print("=== Podsumowanie porównania ===")
print(f"[scikit-learn] Dokładność (k=3) na pełnym zbiorze testowym: {accuracy_sklearn:.4f}")
print(f"[Custom k-NN]  Dokładność (k=3) na {subset_size}-elementowym podzbiorze: {accuracy_custom:.4f}")

import unittest
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def my_knn_predict(X_train, y_train, X_test, k=3):
    predictions = np.zeros(X_test.shape[0], dtype=int)
    for i in range(X_test.shape[0]):
        distances = np.sqrt(np.sum((X_train - X_test[i]) ** 2, axis=1))
        k_smallest_indices = np.argsort(distances)[:k]
        k_nearest_labels = y_train[k_smallest_indices]
        predicted_label = np.argmax(np.bincount(k_nearest_labels))
        predictions[i] = predicted_label
    return predictions

class TestKNN(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        cls.X = np.random.rand(500, 10)  # Zwiększona liczba próbek do 500
        cls.y = np.random.randint(0, 2, 500)  # Klasy binarne (0 lub 1)
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(
            cls.X, cls.y, test_size=0.2, random_state=42
        )
        cls.scaler = StandardScaler()
        cls.X_train = cls.scaler.fit_transform(cls.X_train)
        cls.X_test = cls.scaler.transform(cls.X_test)

    def test_sklearn_knn(self):
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(self.X_train, self.y_train)
        y_pred = knn.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"Dokładność scikit-learn k-NN: {accuracy:.4f}")
        self.assertGreater(accuracy, 0.5, "Dokładność powinna być większa niż 50%")

    def test_custom_knn(self):
        y_pred_custom = my_knn_predict(self.X_train, self.y_train, self.X_test, k=3)
        accuracy = accuracy_score(self.y_test, y_pred_custom)
        print(f"Dokładność własnej implementacji k-NN: {accuracy:.4f}")
        self.assertGreater(accuracy, 0.5, "Dokładność powinna być większa niż 50%")

    def test_knn_identical_samples(self):
        """Test sprawdzający, czy model poprawnie klasyfikuje identyczne próbki"""
        X_train = np.array([[1, 1], [2, 2], [3, 3]])
        y_train = np.array([0, 1, 1])
        X_test = np.array([[1, 1]])
        y_pred = my_knn_predict(X_train, y_train, X_test, k=1)
        self.assertEqual(y_pred[0], 0, "Model powinien poprawnie klasyfikować identyczne próbki")

    def test_knn_different_k_values(self):
        """Test dla różnych wartości k"""
        for k in [1, 3, 5]:
            y_pred = my_knn_predict(self.X_train, self.y_train, self.X_test, k=k)
            accuracy = accuracy_score(self.y_test, y_pred)
            print(f"Dokładność dla k={k}: {accuracy:.4f}")
            self.assertGreater(accuracy, 0.5, f"Dokładność dla k={k} powinna być większa niż 50%")

if __name__ == "__main__":
    unittest.main()

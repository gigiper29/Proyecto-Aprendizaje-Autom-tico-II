"""Estimadores propios para clasificación (BD_IPSA_1940)."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier


class KNNPriorCorregido(ClassifierMixin, BaseEstimator):
    """KNN con corrección por priors: el análogo de `class_weight="balanced"` de la logística.

    Divide la probabilidad de cada clase por su frecuencia en el train del ajuste y predice la clase con el mayor
    cociente. Así una clase minoritaria (p. ej. "Bajo", ~22 %) no queda en desventaja solo por ser menos frecuente
    entre los vecinos. Las frecuencias se calculan en `fit`, así que dentro de la CV salen del train de cada fold.
    """

    def __init__(self, n_neighbors=5, weights="uniform"):
        self.n_neighbors = n_neighbors
        self.weights = weights

    def fit(self, X, y):
        self.knn_ = KNeighborsClassifier(n_neighbors=self.n_neighbors, weights=self.weights).fit(X, y)
        self.classes_ = self.knn_.classes_
        frecuencias = np.array([np.mean(np.asarray(y) == c) for c in self.classes_])
        self.priors_ = frecuencias
        return self

    def predict_proba(self, X):
        return self.knn_.predict_proba(X)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X) / self.priors_, axis=1)]

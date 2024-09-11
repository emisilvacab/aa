# IMPLEMENTANDO NAIVE BAYES

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_curve, classification_report, accuracy_score
from sklearn.feature_selection import SelectKBest, chi2
import matplotlib.pyplot as plt

# Funciones del Naive Bayes implementado
def fit_naive_bayes(X, y, m):
    """Entrena un modelo Naive Bayes basado en la distribución Gaussiana, ajustando suavizado con m"""
    classes = np.unique(y)
    priors = {c: np.mean(y == c) for c in classes}  # P(clase)

    # Calcular la media y varianza para cada característica dado cada clase, ajustando con m
    likelihoods = {
        c: {
            'mean': np.mean(X[y == c], axis=0),
            'var': np.var(X[y == c], axis=0) + (1 / m)  # Ajustar suavizado con m
        }
        for c in classes
    }
    return priors, likelihoods, classes

def gaussian_probability(x, mean, var):
    """Calcula la probabilidad de una característica usando la distribución gaussiana"""
    coeff = 1.0 / np.sqrt(2.0 * np.pi * var)
    exponent = np.exp(-(np.power(x - mean, 2) / (2 * var)))
    return coeff * exponent

def predict_single(x, priors, likelihoods, classes):
    """Predice la clase para una sola muestra x"""
    posteriors = {}

    # Calcular la probabilidad posterior para cada clase
    for c in classes:
        prior = np.log(priors[c])  # Usar log para evitar underflow
        likelihood = np.sum(
            np.log(gaussian_probability(x, likelihoods[c]['mean'], likelihoods[c]['var']))
        )
        posteriors[c] = prior + likelihood

    # Devolver la clase con la mayor probabilidad posterior
    return max(posteriors, key=posteriors.get)

def predict_naive_bayes(X, priors, likelihoods, classes):
    """Predice las clases para un conjunto de muestras"""
    return np.array([predict_single(x, priors, likelihoods, classes) for x in X])

def train_evaluate_naive_bayes(m_value):
    print(f"Entrenando con m={m_value}")

    # Entrenar el modelo Naive Bayes implementado con suavizado (m_value)
    priors, likelihoods, classes = fit_naive_bayes(X_train, y_train, m_value)

    # Predecir
    y_pred = predict_naive_bayes(X_test, priors, likelihoods, classes)

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    print("Matriz de confusión:")
    print(cm)

    # Clasificación (precision, recall, f1-score)
    print("Informe de clasificación:")
    print(classification_report(y_test, y_pred, zero_division=1))

    # Calcular la precisión general
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Precisión: {accuracy}")

    # Curva precision-recall
    y_scores = predict_naive_bayes(X_test, priors, likelihoods, classes)
    precision, recall, _ = precision_recall_curve(y_test, y_scores)

    plt.plot(recall, precision, marker='.')
    plt.title(f"Curva Precision-Recall para Naive Bayes Implementado con m={m_value}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.show()

# 1. Cargar el dataset
dataset = pd.read_csv('./lab1_dataset.csv')
X = dataset.drop('cid', axis=1)  # cid es la columna objetivo
y = dataset['cid']

# 2. Selección de características
k_best = SelectKBest(score_func=chi2, k=10)
X_new = k_best.fit_transform(X, y)

# 3. División del dataset en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=42)

# 4. Entrenar y evaluar con diferentes valores de m
for m in [1, 10, 100, 1000]:
    train_evaluate_naive_bayes(m)

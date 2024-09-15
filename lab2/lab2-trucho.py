# USANDO NAIVE BAYES DE LIBRERIA

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, precision_recall_curve, classification_report, accuracy_score
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, chi2

# 1. Cargar el dataset
# (Usa el dataset que mencionas, puedes ajustar esta parte si tienes alguna particularidad)
dataset = pd.read_csv('./lab1_dataset.csv')
X = dataset.drop('cid', axis=1)  # cid es la columna objetivo
y = dataset['cid']

# 2. Selección de características (feature selection)
# Vamos a seleccionar las mejores K características
# Ajusta k según necesites. Puedes aplicar otras técnicas también.
k_best = SelectKBest(score_func=chi2, k=10)
X_new = k_best.fit_transform(X, y)

# 3. División del dataset en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=42)

# 4. Función para entrenar y evaluar Naive Bayes con distintos valores de m (tamaño equivalente de muestra)
def train_evaluate_naive_bayes(m_value):
    print(f"Entrenando con m={m_value}")

    # Inicializar el clasificador Naive Bayes con suavizado Laplaciano ajustado con m_value
    model = GaussianNB(var_smoothing=1/m_value)

    # Entrenar el modelo
    model.fit(X_train, y_train)

    # Evaluar el modelo
    y_pred = model.predict(X_test)

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    print("Matriz de confusión:")
    print(cm)

    # Clasificación (precision, recall, f1-score)
    print("Informe de clasificación:")
    print(classification_report(y_test, y_pred, zero_division=1))

    # Curva precision-recall
    y_scores = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_scores)

    plt.plot(recall, precision, marker='.')
    plt.title(f"Curva Precision-Recall para m={m_value}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.show()

# 5. Entrenar y evaluar con diferentes valores de m
for m in [1, 10, 100, 1000]:
    train_evaluate_naive_bayes(m)

# 6. Validación cruzada
# Evaluar con validación cruzada (5 folds) para medir estabilidad
model = GaussianNB(var_smoothing=1/10)  # Puedes cambiar por otros valores de m
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"Validación cruzada (5-folds): {scores}")
print(f"Precisión promedio en validación cruzada: {np.mean(scores)}")

import pandas as pd
from sklearn.model_selection import train_test_split
from lab4 import *

####################################################################################################################

# 1. Cargar el dataset y categorizar atributos continuos
dataset = pd.read_csv("./lab1_dataset.csv")
preprocessed_dataset = dataset.drop(["cid", "pidnum", "time"], axis=1)
target_column = dataset["cid"]

# Dividir en 80% entrenamiento y 20% test
dataset_train_full, dataset_test, target_train_full, target_test = train_test_split(
    preprocessed_dataset,
    target_column,
    test_size=0.2,
    random_state=42,
    stratify=target_column,
)

# Separar 10% del conjunto de entrenamiento completo para validación
dataset_train, dataset_val, target_train, target_val = train_test_split(
    dataset_train_full,
    target_train_full,
    test_size=0.1,
    random_state=42,
    stratify=target_train_full,
)

# Mostrar los tamaños de los conjuntos
print(f"Tamaño del conjunto de datos completo: {preprocessed_dataset.shape[0]}")
print(f"Tamaño del conjunto de entrenamiento completo: {dataset_train_full.shape[0]}")
print(f"Tamaño del conjunto de entrenamiento: {dataset_train.shape[0]}")
print(f"Tamaño del conjunto de validación: {dataset_val.shape[0]}")
print(f"Tamaño del conjunto de test: {dataset_test.shape[0]}")

# 2. Modelo de Regresión Logística

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Inicializar el escalador
scaler = StandardScaler()

# Ajustar el escalador y transformar los conjuntos de entrenamiento y validación
dataset_train_scaled = scaler.fit_transform(dataset_train)
dataset_val_scaled = scaler.transform(dataset_val)

# Inicializar el modelo de regresión logística
modelo_logistico = LogisticRegression(max_iter=1000, random_state=42)

# Entrenar el modelo con el conjunto de entrenamiento
modelo_logistico.fit(dataset_train_scaled, target_train)

# Realizar predicciones en el conjunto de validación
target_pred_val = modelo_logistico.predict(dataset_val_scaled)

# Calcular la accuracy en el conjunto de validación
accuracy_val = accuracy_score(target_val, target_pred_val)
print(f"Accuracy en el conjunto de validación: {accuracy_val:.4f}")

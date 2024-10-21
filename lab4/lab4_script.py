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

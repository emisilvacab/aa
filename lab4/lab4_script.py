import pandas as pd
from sklearn.model_selection import train_test_split

# from lab4 import *

####################################################################################################################

# 1. Cargar el dataset
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

# 2. Modelo 1: Regresión Logística

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

# 3. Modelo 2: Red neuronal de una neurona lineal (sin función de activación), con dos salidas, una
# para cada clase objetivo posible del conjunto de datos

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Convertir los datos a tensores
dataset_train_tensor = torch.FloatTensor(dataset_train_scaled)
target_train_tensor = torch.LongTensor(
    target_train.values
)  # Asegúrate de que 'target_train' sea un arreglo de enteros
dataset_val_tensor = torch.FloatTensor(dataset_val_scaled)
target_val_tensor = torch.LongTensor(target_val.values)


# Definir el modelo
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.linear = nn.Linear(dataset_train_tensor.shape[1], 2)  # Asumiendo 2 clases

    def forward(self, x):
        return self.linear(x)


# Inicializar el modelo, la función de pérdida y el optimizador
model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Variables para almacenar la pérdida y precisión
train_loss_history = []
val_loss_history = []
train_accuracy_history = []
val_accuracy_history = []

# Entrenamiento del modelo
num_epochs = 100
for epoch in range(num_epochs):
    # Entrenamiento
    model.train()
    optimizer.zero_grad()
    outputs = model(dataset_train_tensor)
    loss = criterion(outputs, target_train_tensor)
    loss.backward()
    optimizer.step()

    # Calcular precisión de entrenamiento
    _, predicted_train = torch.max(outputs.data, 1)
    train_accuracy = (predicted_train == target_train_tensor).sum().item() / len(
        target_train_tensor
    )

    # Validación
    model.eval()
    with torch.no_grad():
        val_outputs = model(dataset_val_tensor)
        val_loss = criterion(val_outputs, target_val_tensor)

        # Calcular precisión de validación
        _, predicted_val = torch.max(val_outputs.data, 1)
        val_accuracy = (predicted_val == target_val_tensor).sum().item() / len(
            target_val_tensor
        )

    # Guardar estadísticas
    train_loss_history.append(loss.item())
    val_loss_history.append(val_loss.item())
    train_accuracy_history.append(train_accuracy)
    val_accuracy_history.append(val_accuracy)

    # Imprimir información de cada época
    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss.item():.4f}, Val Accuracy: {val_accuracy:.4f}"
        )

## 3.6. Graficar la pérdida y la precisión
plt.figure(figsize=(12, 5))

## Gráfica de la pérdida
plt.subplot(1, 2, 1)
plt.plot(train_loss_history, label="Train Loss")
plt.plot(val_loss_history, label="Validation Loss")
plt.title("Loss vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

## Gráfica de la precisión
plt.subplot(1, 2, 2)
plt.plot(train_accuracy_history, label="Train Accuracy")
plt.plot(val_accuracy_history, label="Validation Accuracy")
plt.title("Accuracy vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.show()

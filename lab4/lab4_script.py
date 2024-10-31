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

# Paso 2: Definir una función para crear y entrenar el modelo con múltiples capas ocultas
def train_model(input_size, hidden_layers, output_size, activation_fn, loss_fn, optimizer_fn, num_epochs=100):
    # Convertir los datos a tensores de PyTorch
    X_train_tensor = torch.tensor(dataset_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(target_train.values, dtype=torch.float32 if output_size == 1 else torch.long)
    X_val_tensor = torch.tensor(dataset_val_scaled, dtype=torch.float32)
    y_val_tensor = torch.tensor(target_val.values, dtype=torch.float32 if output_size == 1 else torch.long)

    # Definir el modelo
    class CustomModel(nn.Module):
        def __init__(self):
            super(CustomModel, self).__init__()
            layers = []

            # Capa de entrada
            last_size = input_size

            # Capas ocultas
            for hidden_size in hidden_layers:
                layers.append(nn.Linear(last_size, hidden_size))
                if activation_fn:  # Añadir activación solo si no es None
                    layers.append(activation_fn)
                last_size = hidden_size

            # Capa de salida (sin activación, se usa sigmoide después)
            layers.append(nn.Linear(last_size, output_size))
            self.model = nn.Sequential(*layers)

        def forward(self, x):
            x = self.model(x)
            if output_size == 1:
                x = torch.sigmoid(x) # La salida es una probabilidad
            return x

    model = CustomModel()

    # Definir la función de pérdida y el optimizador
    criterion = loss_fn
    optimizer = optimizer_fn(model.parameters())

    # Almacenar pérdidas y accuracy
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    # Entrenamiento del modelo
    for epoch in range(num_epochs):
        # Modo de entrenamiento
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        # Ajustar la función de pérdida según el tipo de salida
        if output_size == 1:  # Clasificación binaria
            loss = criterion(outputs, y_train_tensor.view(-1, 1))
            # Convertir salidas a 0 o 1 para la accuracy
            train_predictions = (outputs > 0.5).float()
        else:  # Clasificación multiclase
            loss = criterion(outputs, y_train_tensor)
            # Usar argmax para obtener la clase con mayor probabilidad
            train_predictions = torch.argmax(outputs, dim=1)
        loss.backward()
        optimizer.step()

        # Evaluación en el conjunto de validación
        model.eval()
        with torch.no_grad():
            train_loss = loss.item()
            val_outputs = model(X_val_tensor)
            if output_size == 1:
                val_loss = criterion(val_outputs, y_val_tensor.view(-1, 1)).item()
                val_predictions = (val_outputs > 0.5).float()
            else:
                val_loss = criterion(val_outputs, y_val_tensor).item()
                val_predictions = torch.argmax(val_outputs, dim=1)

            # Calcular la accuracy
            train_accuracy = (train_predictions.squeeze() == y_train_tensor).float().mean().item()
            val_accuracy = (val_predictions.squeeze() == y_val_tensor).float().mean().item()

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)

    print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Training Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}')

    # Graficar la pérdida y la accuracy
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()

    return model

# Paso 3: Ejemplo de uso para entrenar el modelo con múltiples capas ocultas
# Modelo 2
input_size = dataset_train_scaled.shape[1] # Número de características de entrada
hidden_layers = [] # No hay capas ocultas
output_size = 2 # Ya que hay dos salidas (una para cada clase).
activation_fn = None # Porque no hay una función de activación después de la capa de salida
loss_fn = nn.CrossEntropyLoss()
optimizer_fn = lambda params: torch.optim.SGD(params, lr=0.01, momentum=0.9)
num_epochs = 100

# Entrenar y evaluar el modelo
trained_model2 = train_model(input_size, hidden_layers, output_size, activation_fn, loss_fn, optimizer_fn, num_epochs)
del optimizer_fn, loss_fn

# Modelo 3
input_size = dataset_train_scaled.shape[1]  # Número de características de entrada
hidden_layers = [] # No hay capas ocultas
output_size = 1 # La salida es la probabilidad de una de las clases
activation_fn = nn.Sigmoid()
loss_fn = nn.BCELoss()
optimizer_fn = lambda params: torch.optim.SGD(params, lr=0.01, momentum=0.9)
num_epochs = 100

# Entrenar y evaluar el modelo
trained_model3 = train_model(input_size, hidden_layers, output_size, activation_fn, loss_fn, optimizer_fn, num_epochs)
del optimizer_fn, loss_fn

# Modelo 4
input_size = dataset_train_scaled.shape[1]  # Número de características de entrada
hidden_layers = [16]  # Aquí se puede agregar más capas, por ejemplo, [16, 32] para dos capas ocultas
output_size = 1  # La salida es la probabilidad de una de las clases
activation_fn = nn.Sigmoid()
loss_fn = nn.BCELoss()
optimizer_fn = lambda params: torch.optim.Adam(params, lr=0.01)
num_epochs = 100

# Entrenar y evaluar el modelo
trained_model4 = train_model(input_size, hidden_layers, output_size, activation_fn, loss_fn, optimizer_fn, num_epochs)
del optimizer_fn, loss_fn

learning_rates = [0.001, 0.01, 0.1]

# Modelo 5.1
input_size = dataset_train_scaled.shape[1]  # Número de características de entrada
hidden_layers = [16]  # Aquí se puede agregar más capas, por ejemplo, [16, 32] para dos capas ocultas
output_size = 1  # La salida es la probabilidad de una de las clases
activation_fn = None
loss_fn = nn.CrossEntropyLoss()
optimizer_fn = lambda params: torch.optim.Adam(params, lr=0.01)
num_epochs = 100

for lr in learning_rates:
    print("-------------------------------------------------------------------------------------------")
    print(f"Entrenando modelo 5.1 con tasa de aprendizaje: {lr}")
    optimizer_fn = lambda params: torch.optim.Adam(params, lr=lr)
    trained_model51 = train_model(input_size, hidden_layers, output_size, activation_fn, loss_fn, optimizer_fn, num_epochs)

del optimizer_fn, loss_fn

# Modelo 5.2
input_size = dataset_train_scaled.shape[1]  # Número de características de entrada
hidden_layers = [16, 16]  # Aquí se puede agregar más capas, por ejemplo, [16, 32] para dos capas ocultas
output_size = 2
activation_fn = None
loss_fn = nn.CrossEntropyLoss()
optimizer_fn = lambda params: torch.optim.SGD(params, lr=0.01, momentum=0.9)
num_epochs = 100

for lr in learning_rates:
    print("-------------------------------------------------------------------------------------------")
    print(f"Entrenando modelo 5.2 con tasa de aprendizaje: {lr}")
    optimizer_fn = lambda params: torch.optim.SGD(params, lr=lr, momentum=0.9)
    trained_model52 = train_model(input_size, hidden_layers, output_size, activation_fn, loss_fn, optimizer_fn, num_epochs)

del optimizer_fn, loss_fn

# Modelo 5.3
input_size = dataset_train_scaled.shape[1]  # Número de características de entrada
hidden_layers = []  # Aquí se puede agregar más capas, por ejemplo, [16, 32] para dos capas ocultas
output_size = 2
activation_fn = nn.Sigmoid()
loss_fn = nn.CrossEntropyLoss()
optimizer_fn = lambda params: torch.optim.SGD(params, lr=0.01, momentum=0.9)
num_epochs = 100

for lr in learning_rates:
    print("-------------------------------------------------------------------------------------------")
    print(f"Entrenando modelo 5.3 con tasa de aprendizaje: {lr}")
    optimizer_fn = lambda params: torch.optim.SGD(params, lr=lr, momentum=0.9)
    trained_model53 = train_model(input_size, hidden_layers, output_size, activation_fn, loss_fn, optimizer_fn, num_epochs)

del optimizer_fn, loss_fn

del trained_model2, trained_model3, trained_model4, trained_model51, trained_model52, trained_model53
torch.cuda.empty_cache()  # Si estás usando una GPU

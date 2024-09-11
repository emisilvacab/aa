# IMPLEMENTANDO NAIVE BAYES

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_curve, classification_report, accuracy_score
from sklearn.feature_selection import SelectKBest, chi2
import matplotlib.pyplot as plt

# Funciones del Naive Bayes implementado
def fit_naive_bayes(dataset_train, target_train, m):
	"""
	Entrena un modelo Naive Bayes basado en la distribución Gaussiana, ajustando suavizado con m.

	Args:
		dataset_train (numpy.ndarray o pandas.DataFrame): Conjunto de características de entrenamiento (muestras x características).
		target_train (numpy.ndarray o pandas.Series): Vector de etiquetas o clases correspondiente al conjunto de entrenamiento.
		m (int o float): Valor de suavizado (Laplace smoothing) para ajustar la varianza calculada.

	Returns:
		priors (dict): Diccionario que contiene las probabilidades a priori P(clase) para cada clase.
		likelihoods (dict): Diccionario de medias y varianzas para cada característica dada una clase.
		classes (numpy.ndarray): Array con las clases únicas en el conjunto de datos.
	"""
	# encontrar las clases unicas en el conjunto de entrenamiento
	classes = np.unique(target_train)
	# calcular todas las probabilidades P(clase) de cada clase (proporcion de muestras de cada clase)
	priors = {c: np.mean(target_train == c) for c in classes}

	# Calcular la media y varianza para cada característica dado cada clase, ajustando con m
	likelihoods = {
			c: {
					'mean': np.mean(dataset_train[target_train == c], axis=0),
					'var': np.var(dataset_train[target_train == c], axis=0) + (1 / m)  # Ajustar suavizado con m
			}
			for c in classes
	}
	return priors, likelihoods, classes

def gaussian_probability(x, mean, var):
	"""
	Calcula la probabilidad de una característica utilizando la distribución Gaussiana.

	Args:
		x (float o numpy.ndarray): Valor de la característica que se quiere evaluar.
		mean (float o numpy.ndarray): Media de la característica dada la clase.
		var (float o numpy.ndarray): Varianza de la característica dada la clase (ajustada con suavizado).

	Returns:
		float o numpy.ndarray: Probabilidad de que el valor x pertenezca a la distribución con la media y varianza dadas.
	"""
	coeff = 1.0 / np.sqrt(2.0 * np.pi * var)
	exponent = np.exp(-(np.power(x - mean, 2) / (2 * var)))
	return coeff * exponent

def predict_single(x, priors, likelihoods, classes):
	"""
	Predice la clase para una sola muestra utilizando el modelo Naive Bayes.

	Args:
		x (numpy.ndarray): Muestra (vector de características) para la cual se va a predecir la clase.
		priors (dict): Diccionario de probabilidades a priori P(clase) para cada clase.
		likelihoods (dict): Diccionario de medias y varianzas para cada característica dada una clase.
		classes (numpy.ndarray): Array con las clases únicas del modelo.

	Returns:
		clase_predicha (int o str): La clase con la mayor probabilidad posterior para la muestra dada.
	"""
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

def predict_naive_bayes(dataset_train, priors, likelihoods, classes):
	"""
	Predice las clases para un conjunto de muestras utilizando Naive Bayes.

	Args:
		dataset_train (numpy.ndarray o pandas.DataFrame): Conjunto de muestras (características) que se desean clasificar.
		priors (dict): Diccionario de probabilidades a priori P(clase) para cada clase.
		likelihoods (dict): Diccionario de medias y varianzas para cada característica dada una clase.
		classes (numpy.ndarray): Array con las clases únicas del modelo.

	Returns:
			numpy.ndarray: Array con las clases predichas para cada muestra en el conjunto de datos.
	"""
	return np.array([predict_single(x, priors, likelihoods, classes) for x in dataset_train])

def train_evaluate_naive_bayes(m):
	"""
	Entrena y evalúa el modelo Naive Bayes utilizando el suavizado especificado y muestra los resultados de la evaluación.

	Args:
		m (int o float): Valor de suavizado (Laplace smoothing) utilizado para ajustar la varianza en el modelo.

	Returns:
		None: Imprime la matriz de confusión, el informe de clasificación y la precisión, y genera una curva Precision-Recall.
	"""
	print(f"Entrenando con m={m}")

	# Entrenar el modelo Naive Bayes implementado con suavizado (m)
	priors, likelihoods, classes = fit_naive_bayes(dataset_train, target_train, m)

	# Predecir
	target_pred = predict_naive_bayes(dataset_test, priors, likelihoods, classes)

	# Matriz de confusión
	cm = confusion_matrix(target_test, target_pred)
	print("Matriz de confusión:")
	print(cm)

	# Clasificación (precision, recall, f1-score)
	print("Informe de clasificación:")
	print(classification_report(target_test, target_pred, zero_division=1))

	# Calcular la precisión general
	accuracy = accuracy_score(target_test, target_pred)
	print(f"Precisión: {accuracy}")

	# Curva precision-recall
	target_scores = predict_naive_bayes(dataset_test, priors, likelihoods, classes)
	precision, recall, _ = precision_recall_curve(target_test, target_scores)

	plt.plot(recall, precision, marker='.')
	plt.title(f"Curva Precision-Recall para Naive Bayes Implementado con m={m}")
	plt.xlabel("Recall")
	plt.ylabel("Precision")
	plt.show()

# 1. Cargar el dataset
dataset = pd.read_csv('./lab1_dataset.csv')
preprocessed_dataset = dataset.drop('cid', axis=1).drop('pidnum', axis=1)  # cid es la columna objetivo
target_column = dataset['cid']

# 2. Selección de características
k_best = SelectKBest(score_func=chi2, k=10)
preprocessed_dataset = k_best.fit_transform(preprocessed_dataset, target_column)

# 3. División del dataset en conjunto de entrenamiento y prueba
dataset_train, dataset_test, target_train, target_test = train_test_split(preprocessed_dataset, target_column, test_size=0.2, random_state=42)

# 4. Entrenar y evaluar con diferentes valores de m
for m in [1, 10, 100, 1000]:
  train_evaluate_naive_bayes(m)

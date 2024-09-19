import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, precision_recall_curve, classification_report, accuracy_score
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.utils.estimator_checks import check_estimator
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import PrecisionRecallDisplay

class NaiveBayes():
	def __init__(self, m):
		self.m = m

	def fit(self, dataset_train, target_train):
		"""
		Entrena un modelo Naive Bayes basado en la distribución multinomial, ajustando suavizado con m.

		Args:
			dataset_train (numpy.ndarray o pandas.DataFrame): Conjunto de características de entrenamiento (muestras x características).
			target_train (numpy.ndarray o pandas.Series): Vector de etiquetas o clases correspondiente al conjunto de entrenamiento.

		Returns:
			self (NaiveBayes) con los atributos:
				priors (dict): Diccionario que contiene las probabilidades a priori P(clase) para cada clase.
				likelihoods (dict): Diccionario con los conteos y suavizado de cada característica dado una clase (probabilidades condicionales).
				classes (numpy.ndarray): Array con las clases únicas en el conjunto de datos.
		"""
		# encontrar las clases unicas en el conjunto de entrenamiento
		self.classes = np.unique(target_train)

		# calcular todas las probabilidades P(clase) de cada clase (proporcion de muestras de cada clase)
		# {0: proporcion de 0's, 1: proporcion de 1's}
		self.priors = {c: np.mean(target_train == c) for c in self.classes}

		# Inicializar el diccionario para las probabilidades condicionales (likelihoods)
		self.likelihoods = {}

		for c in self.classes:
			if isinstance(dataset_train, pd.DataFrame):
				subset = dataset_train[target_train == c].to_numpy()
			else:
				subset = dataset_train[target_train == c]

			# Calcular n: número total de instancias en la clase c
			n = len(subset)

			# Almacenar probabilidades condicionales para cada característica
			likelihoods_for_class = {}

			if isinstance(dataset_train, pd.DataFrame):
				dataset_train_array = dataset_train[target_train == c].to_numpy()
			else:
				dataset_train_array = dataset_train[target_train == c]

			for col in range(dataset_train_array.shape[1]):
				unique_values = np.unique(dataset_train_array[:, col])

				# Obtener la cantidad de valores posibles para esta característica (cantidad de valores únicos)
				num_unique_values = len(unique_values)

				# Probabilidad a priori de la característica
				p = 1 / num_unique_values

				# Contar la cantidad de instancias para cada valor de la característica dado la clase c
				e = {value: np.sum(subset[:, col] == value) for value in unique_values}

				# Aplicando la fórmula del m-estimador
				likelihood = {value: (e[value] + self.m * p) / (n + self.m) for value in unique_values}

				# Almacenar la probabilidad condicional para esta característica
				likelihoods_for_class[col] = likelihood

			# Almacenar las probabilidades condicionales para la clase c
			self.likelihoods[c] = likelihoods_for_class

		return self

	def predict(self, dataset_test):
		"""
		Predice las clases para un conjunto de muestras utilizando Naive Bayes.

		Args:
			dataset_test (numpy.ndarray o pandas.DataFrame): Conjunto de muestras (características) que se desean clasificar.

		Returns:
				numpy.ndarray: Array con las clases predichas para cada muestra en el conjunto de datos.
		"""
		return np.array([self.predict_single(x) for x in dataset_test])

	def predict_single(self, x):
		"""
		Predice la clase para una sola muestra utilizando el modelo Naive Bayes.

		Args:
			x (numpy.ndarray): Muestra (vector de características) para la cual se va a predecir la clase.
			self (NaiveBayes) con los atributos:
				priors (dict): Diccionario de probabilidades a priori P(clase) para cada clase.
				likelihoods (dict): Diccionario de probabilidades de características para cada clase.
				classes (numpy.ndarray): Array con las clases únicas del modelo.

		Returns:
			clase_predicha (int o str): La clase con la mayor probabilidad posterior para la muestra dada.
		"""
		posteriors = {}

		# Calcular la probabilidad posterior para cada clase
		for c in self.classes:
			# Iniciar con la probabilidad a priori de la clase
			prior = np.log(self.priors[c])  # Usar log para evitar underflow

			# Inicializar la suma de los logaritmos de las probabilidades condicionales
			likelihood = 0

			for feature in range(len(x)):
				feature_value = x[feature]
				feature_likelihood = self.likelihoods[c].get(feature, {})
				feature_value_likelihood = feature_likelihood.get(feature_value, 1e-9)
				likelihood += np.log(feature_value_likelihood)

			posteriors[c] = prior + likelihood

		# Devolver la clase con la mayor probabilidad posterior
		return max(posteriors, key=posteriors.get)

	def get_params(self, deep=True):
		# Devuelve un diccionario con los parámetros del estimador
		return {'m': self.m}

def train_evaluate_naive_bayes(m, dataset_train, target_train, dataset_test):
	"""
	Entrena y evalúa el modelo Naive Bayes utilizando el suavizado especificado y muestra los resultados de la evaluación.

	Args:
		m (int o float): Valor de suavizado (Laplace smoothing) utilizado para ajustar la varianza en el modelo.

	Returns:
		None: Imprime la matriz de confusión, el informe de clasificación y la precisión, y genera una curva Precision-Recall.
	"""
	print(f"Entrenando con m={m}")

  # Inicializar el clasificador Naive Bayes con suavizado Laplaciano ajustado con m
	model = NaiveBayes(m)

	# Entrenar el modelo Naive Bayes implementado con suavizado (m)
	model.fit(dataset_train, target_train)

	# Predecir
	target_pred = model.predict(dataset_test)

	return target_pred

# IMPLEMENTANDO NAIVE BAYES

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
		# [0, 1]
		self.classes = np.unique(target_train)
		# calcular todas las probabilidades P(clase) de cada clase (proporcion de muestras de cada clase)
		# {0: proporcion de 0's, 1: proporcion de 1's}
		self.priors = {c: np.mean(target_train == c) for c in self.classes}

		# Inicializar el diccionario para las probabilidades condicionales (likelihoods)
		self.likelihoods = {}

		# Para cada clase, calcular la probabilidad condicional de cada característica
		for c in self.classes:
			# Filtrar las filas de dataset_train donde la clase es igual a c
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

			# Para cada característica (columna)
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
			## comentarios con ## hace lo mismo sin log
			## prior = self.priors[c]

			# Inicializar la suma de los logaritmos de las probabilidades condicionales
			likelihood = 0
			## likelihood = priors

			# Iterar sobre cada característica
			for feature in range(len(x)):
				feature_value = x[feature]
				feature_likelihood = self.likelihoods[c].get(feature, {})
				feature_value_likelihood = feature_likelihood.get(feature_value, 1e-9)

				likelihood += np.log(feature_value_likelihood)
				## likelihood *= feature_value_likelihood

			posteriors[c] = prior + likelihood
			## posteriors[c] = likelihood

		# Devolver la clase con la mayor probabilidad posterior
		return max(posteriors, key=posteriors.get)

	def get_params(self, deep=True):
		# Devuelve un diccionario con los parámetros del estimador
		return {'m': self.m}

#################################################### SCRIPT ####################################################

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

def categorize_numeric_features(df, bins=3):
    """
    Convierte atributos numéricos en categóricos utilizando bins.
    """
    ATTRIBUTES_REQUIRING_RANGES = ['time', 'age', 'wtkg', 'karnof', 'preanti', 'cd40', 'cd420', 'cd80', 'cd820']
    df_categorized = df.copy()

    for column in ATTRIBUTES_REQUIRING_RANGES:
        # Convertir el atributo numérico en categórico usando `pd.cut`
        df_categorized[column], bins_used = pd.cut(
            df_categorized[column],
            bins=bins,
            labels=False,
            retbins=True
        )

    return df_categorized

# 1. Cargar el dataset y categorizar atributos continuos
dataset = pd.read_csv('./lab1_dataset.csv')
preprocessed_dataset = dataset.drop(['cid', 'pidnum'], axis=1)
target_column = dataset['cid']
preprocessed_dataset = categorize_numeric_features(preprocessed_dataset)

# 2. División del dataset en conjunto de entrenamiento y prueba
class_proportions = target_column.value_counts(normalize=True)
print("Las proporciones de valores de la clase objetivo son:")
print(class_proportions)

dataset_train, dataset_test, target_train, target_test = train_test_split(
	preprocessed_dataset, target_column, test_size=0.2, random_state=42, stratify=target_column
)
dataset_test = dataset_test.to_numpy()

# 3. Aplicar la técnica de Chi-2 (seleccion de atributos)
chi2_selector = SelectKBest(chi2, k=11)  # Seleccionar los 10 mejores atributos
dataset_train_with_selected_attributes = chi2_selector.fit_transform(dataset_train, target_train)
selected_columns = dataset_train.columns[chi2_selector.get_support()]
print(f"Los atributos seleccionados por chi-2 son: {selected_columns}")

# 4. Calcular la matriz de correlación y eliminar atributos con una relacion mayor a 0.85
dataset_train_df = pd.DataFrame(dataset_train_with_selected_attributes, columns=selected_columns)
correlation_matrix = dataset_train_df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()

threshold = 0.85
columns_to_discard = set()
for i in range(len(correlation_matrix.columns)):
  for j in range(i):
    if abs(correlation_matrix.iloc[i, j]) > threshold:
      columns_to_discard.add(correlation_matrix.columns[i])
print(f"Atributos descartados por alta correlación: {list(columns_to_discard)}")

dataset_train = dataset_train_df.drop(columns=columns_to_discard).to_numpy()

# 5. Entrenar y evaluar con diferentes valores de m
for m in [1, 10, 100, 1000]:
  target_pred = train_evaluate_naive_bayes(m, dataset_train, target_train, dataset_test)

	# Curva precision-recall
  display = PrecisionRecallDisplay.from_predictions(
    target_test, target_pred, name="LinearSVC", plot_chance_level=True
  )
  precision, recall, _ = precision_recall_curve(target_test, target_pred)

  plt.plot(recall, precision, marker='.')
  plt.title(f"Curva Precision-Recall para Naive Bayes Implementado con m={m}")
  plt.xlabel("Recall")
  plt.ylabel("Precision")
  plt.show()

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

  modelCross = NaiveBayes(m)
  scores = cross_val_score(modelCross, dataset_train, target_train, cv=5, scoring='accuracy')
  print(f"Validación cruzada (5-folds): {scores}")
  print(f"Precisión promedio en validación cruzada: {np.mean(scores)}")
  print("--------------------------------------------------------------------------------------------")

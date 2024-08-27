import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler  # O cualquier método de preprocesamiento
import matplotlib.pyplot as plt
import seaborn as sns

"""
dataset = [
  {
    "pidnum": 10056, # id
    "time": 948,
    "cid": 0 # objetivo
  }
]
"""

def __init__():



# 1. Cargar el csv
  data = pd.read_csv('lab1_dataset.csv')

  print(data.head())
  print(data.info())
  print(data.describe())

# 2. Manejo de valores faltantes
#
# La descripcion de los datos nos dice que no hay valores faltantes tambien se ve revisando la data
#
# 3. Codificación de Variables Categóricas
# Variables categóricas como trt, race, gender, entre otras, necesitan ser convertidas en variables numéricas. scikit-learn tiene herramientas como OneHotEncoder o LabelEncoder para este propósito.
#
# Ya estan codificadas
#

# 4. Estandarización de Variables Numéricas
# Es recomendable estandarizar las variables numéricas para que tengan una media de 0 y una desviación estándar de 1.

from sklearn.preprocessing import StandardScaler

# Variables numéricas para estandarizar
numeric_columns = ['age', 'wtkg', 'karnof', 'cd40', 'cd420', 'cd80', 'cd820']

# Estandarización
scaler = StandardScaler()
df_encoded[numeric_columns] = scaler.fit_transform(df_encoded[numeric_columns])

print(df_encoded[numeric_columns].head())

# 5. Preparación de los Datos para el Modelado
# Dividir los datos en características (X) y la variable objetivo (y).
# La columna cid es la variable objetivo según la descripción.

# Separar características y objetivo
X = df_encoded.drop(columns=['cid', 'pidnum'])  # 'cid' es la variable objetivo, 'pidnum' es el identificador
y = df_encoded['cid']

# 6. División del Conjunto de Datos
# Dividir los datos en conjuntos de entrenamiento y prueba para evaluar el rendimiento del modelo.
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Train size: {X_train.shape}")
print(f"Test size: {X_test.shape}")


# 7. Modelado y Evaluación
# Ahora puedes proceder a aplicar el algoritmo ID3 (o cualquier otro algoritmo) sobre el conjunto de datos preprocesado.

# Aquí podrías implementar y entrenar tu modelo usando X_train, y_train
# Luego evaluarías el modelo con X_test, y_test

def id3(dataset, attributes, tree):
  selected_attr = select_attr(dataset, attributes)

  if same_value(dataset, selected_attr):
    leaf_value = dataset[0].get(selected_attr)
  elif len(attributes) == 1:
    leaf_value = most_frequent_attr_value(dataset, selected_attr)
  else:
    # for vi (value) in attribute A:
    # create leaf
    # ejemplos_vi = ej in dataset with ej.get(A) = vi
    # if ejemplos_vi == []:
    # etiqueto con most_frequent_attr_value(dataset, A)
    # else:
    # id3(ejemplos_vi, attributes -- A)


  # if (all atributo_seleccionado in dataset):
    # print("Hello")
# id3()
# crear raiz (seleccionar atributo)

# if todo ejemplo tiene mismo valor -> etiqueto con valor
# if no tengo mas atributos -> etiqueto con valor mas comun
# else:
#     raiz pregunta por atributo A
#     for vi in A:
#       genero rama
#       Ejemplosvi = {ejemplos tal que A = vi}
#       if Ejemplosvi = vacio -> etiqueta con valor mas comun
#       else -> ID3 (Ejemplosvi, Atributos - A)


def same_value(dataset, selected_attr):
  """
  Calculate if all elements have the same value on the selected_attr key

  Parameters:
  dataset (list)
  selected_attr (string)

  Returns:
  bool: All elements have the same value on selected_attr key?

  Example:
  >>> same_value([{"a": 1, "b": 2}, {"a": 1, "b": 3}], "a")
  true
  """
  all(ej.get(selected_attr) == dataset[0].get(selected_attr) for ej in dataset)

def most_frequent_attr_value(attr_list, attribute):
  values_list = [getattr(objeto, attribute) for objeto in attr_list]

  counter = 0
  num = values_list[0]

  for i in values_list:
    curr_frequency = values_list.count(i)
    if(curr_frequency> counter):
      counter = curr_frequency
      num = i

  return num


########################################################
import numpy as np

def entropy(S):
    """
    Calculates the entropy of a set of labels.

    Args:
        S (numpy.ndarray): A vector of class labels.

    Returns:
        float: The entropy of the label set.
    """
    # bincount -> count number of occurrences of each value in array of non-negative ints.
    counts = np.bincount(S)
    probabilities = counts / len(S)

    return -np.sum(probabilities * np.log2(probabilities + 1e-9))

def information_gain(X, S, attribute_index):
    """
    Calculates the information gain from splitting the data based on a attribute.

    Args:
        X (numpy.ndarray): Matrix of attributes.
        S (numpy.ndarray): Vector of class labels.
        attribute_index (int): Index of the attribute to evaluate.

    Returns:
        float: The information gain from splitting on the attribute.
    """
    original_entropy = entropy(S)

    # unique -> returns the sorted unique elements of X
    # return_counts=True: also return the number of times each unique item appears in X
    unique_values, counts = np.unique(X[:, attribute_index], return_counts=True)
    probabilities = counts / len(X)

    return original_entropy - np.sum(probabilities * [entropy(S[X[:, attribute_index] == value]) for value in unique_values])

def best_attribute_to_split(X, S):
    """
    Finds the index of the attribute that provides the highest information gain.

    Args:
        X (numpy.ndarray): Matrix of attributes.
        S (numpy.ndarray): Vector of class labels.

    Returns:
        int: The index of the attribute that provides the highest information gain.
    """
    # shape -> returns the shape of the array
    # we use the index 1 inside the shape to get the number of columns (attributes on this case)
    num_attributes = X.shape[1]
    gains = [information_gain(X, S, i) for i in range(num_attributes)]
    return np.argmax(gains)
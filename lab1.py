
import pandas
import numpy

def id3(dataset, attributes, target_attribute, parent_node_class = None):
  """
  Implementación del algoritmo ID3 para construir un árbol de decisión.
  La primer llamada a id3 debería ser: id3(dataset, attributes, target_attribute)
  La implementación soportada por id3 hace que el target_attribute solo pueda tener valor ser 1 o 0 en el dataset

  Args:
      dataset (pandas.DataFrame): El conjunto de datos de entrada, un DataFrame de pandas que contiene los datos.
      attributes (list): Lista de atributos disponibles para realizar divisiones (sin incluir a target_attribute).
      target_attribute (str): El nombre de la columna que contiene la etiqueta o clase objetivo.
      parent_node_class (str, opcional): La clase del nodo padre, utilizada en caso de que no queden más atributos.

  Returns:
      dict o str: Un árbol de decisión representado como un diccionario anidado, o un valor de clase.
  """

  if len(numpy.unique(dataset[target_attribute])) == 1:
    # Si todos los ejemplos tienen el mismo valor → etiquetar con ese valor
    return str(numpy.unique(dataset[target_attribute])[0])

  elif len(attributes) == 0:
    # Si no me quedan atributos → etiquetar con el valor más común
    return str(parent_node_class)

  else:
    # parent_node_class es usado en caso de que no nos quedan mas atributos en la proxima llamada al id3
    parent_node_class = determine_majority_class(dataset, target_attribute)

    dataset_matrix = dataset[attributes].to_numpy()
    classes_vector = dataset[target_attribute].to_numpy()
    index_best_attribute = best_attribute_to_split(dataset_matrix, classes_vector)
    best_attribute = attributes[index_best_attribute]

    tree = {best_attribute: {}}
    remaining_attributes = attributes.copy()
    remaining_attributes.remove(best_attribute)

    for value in numpy.unique(dataset[best_attribute]):
      sub_data = dataset.where(dataset[best_attribute] == value).dropna()
      subtree = id3(sub_data, remaining_attributes, target_attribute, parent_node_class)
      tree[best_attribute][value] = subtree

    return tree

def determine_majority_class(dataset, target_attribute):
  """
  Determina la clase mayoritaria en el conjunto de datos.

  Args:
      dataset (pandas.DataFrame): El conjunto de datos de entrada.
      target_attribute (str): El nombre de la columna que contiene las etiquetas o clases objetivo.

  Returns:
      str: La clase mayoritaria.
  """
  return numpy.unique(dataset[target_attribute])[numpy.argmax(numpy.unique(dataset[target_attribute], return_counts = True)[1])]

def entropy(S):
  """
  Calcula la entropía de un conjunto de etiquetas.

  Args:
      S (numpy.ndarray): Un vector de etiquetas de clase.

  Returns:
      float: La entropía del conjunto de etiquetas.
  """
  # bincounts -> cuenta el número de ocurrencias de cada valor en el array de etiquetas.
  counts = numpy.bincount(S)
  probabilities = counts / len(S)

  return -numpy.sum(probabilities * numpy.log2(probabilities + 1e-9))

def information_gain(X, S, attribute_index):
  """
  Calcula la ganancia de información al dividir los datos basado en un atributo.

  Args:
      X (numpy.ndarray): Matriz de atributos.
      S (numpy.ndarray): Vector de etiquetas de clase.
      attribute_index (int): Índice del atributo a evaluar.

  Returns:
      float: La ganancia de información al dividir en el atributo.
  """
  original_entropy = entropy(S)

  # unique_values -> encuentra los valores únicos del atributo, y counts su frecuencia (debido a return_counts=True).
  unique_values, counts = numpy.unique(X[:, attribute_index], return_counts = True)
  probabilities = counts / len(X)

  # Resta la entropía ponderada de cada división de atributo de la entropía original para obtener la ganancia de información.
  return original_entropy - numpy.sum(probabilities * [entropy(S[X[:, attribute_index] == value]) for value in unique_values])

def best_attribute_to_split(X, S):
  """
  Encuentra el índice del atributo que proporciona la mayor ganancia de información.

  Args:
      X (numpy.ndarray): Matriz de atributos.
      S (numpy.ndarray): Vector de etiquetas de clase.

  Returns:
      int: El índice del atributo que proporciona la mayor ganancia de información.
  """
  # Número de atributos (columnas) en X.
  num_attributes = X.shape[1]
  # Ganancia de información para cada atributo.
  gains = [information_gain(X, S, i) for i in range(num_attributes)]
  # Índice del atributo con mayor ganancia de información.
  return numpy.argmax(gains)

####################################################################################################

# SCRIPT DE PRUEBA
# TODO: Remover cuando se empiecen a usar el csv real

# Crear un DataFrame de ejemplo
data = {
    'Attribute1': ['A', 'A', 'B', 'B', 'A'],
    'Attribute2': ['X', 'X', 'Y', 'Y', 'Y'],
    'Target': [1, 1, 0, 0, 0]
}
df = pandas.DataFrame(data)

# Guardar el DataFrame en un archivo CSV
df.to_csv('example_data.csv', index=False)

# Cargar el archivo CSV
dataset = pandas.read_csv('example_data.csv')

# Definir los atributos y el atributo objetivo
attributes = ['Attribute1', 'Attribute2']
target_attribute = 'Target'

# Llamar a la función id3 para construir el árbol
tree = id3(dataset, attributes, target_attribute)

# Imprimir el árbol de decisión generado
print(tree)

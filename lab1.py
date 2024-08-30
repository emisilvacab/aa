
import pandas as pd
import numpy as np

# La primer llamada a id3 debería ser: id3(dataset, attributes, target_attribute)
#
# dataset: Es nuestro dataset, un DataFrame de pandas que contiene los datos.
# attributes: Es la lista de atributos que el algoritmo utilizará para dividir
#             los datos (no debe incluir a target_attribute).
# target_attribute: Es el nombre de la columna que contiene la clase o etiqueta que queremos predecir.
#
# La implementación soportada por id3 hace que el target_attribute solo pueda tener valor ser 1 o 0 en el dataset

def id3(dataset, attributes, target_attribute, parent_node_class = None):
  if len(np.unique(dataset[target_attribute])) == 1:
    # Si todos los ejemplos tienen el mismo valor → etiquetar con ese valor
    return str(np.unique(dataset[target_attribute])[0])

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

    for value in np.unique(dataset[best_attribute]):
      sub_data = dataset.where(dataset[best_attribute] == value).dropna()
      subtree = id3(sub_data, remaining_attributes, target_attribute, parent_node_class)
      tree[best_attribute][value] = subtree

    return tree

def determine_majority_class(dataset, target_attribute):
  return np.unique(dataset[target_attribute])[np.argmax(np.unique(dataset[target_attribute], return_counts = True)[1])]

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
  unique_values, counts = np.unique(X[:, attribute_index], return_counts = True)
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

####################################################################################################

# SCRIPT DE PRUEBA
# TODO: Remover cuando se empiecen a usar el csv real

# Crear un DataFrame de ejemplo
data = {
    'Attribute1': ['A', 'A', 'B', 'B', 'A'],
    'Attribute2': ['X', 'X', 'Y', 'Y', 'Y'],
    'Target': [1, 1, 0, 0, 0]
}
df = pd.DataFrame(data)

# Guardar el DataFrame en un archivo CSV
df.to_csv('example_data.csv', index=False)

# Cargar el archivo CSV
dataset = pd.read_csv('example_data.csv')

# Definir los atributos y el atributo objetivo
attributes = ['Attribute1', 'Attribute2']
target_attribute = 'Target'

# Llamar a la función id3 para construir el árbol
tree = id3(dataset, attributes, target_attribute)

# Imprimir el árbol de decisión generado
print(tree)

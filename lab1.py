import pandas
import numpy
from sklearn import ensemble, model_selection, metrics, tree

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
    return str(dataset[target_attribute])

  elif len(attributes) == 0:
    # Si no me quedan atributos → etiquetar con el valor más común
    return str(parent_node_class)

  else:
    # parent_node_class es usado en caso de que no nos quedan mas atributos en la proxima llamada al id3
    parent_node_class = most_common_value(dataset, target_attribute)

    dataset_matrix = dataset[attributes].to_numpy()
    classes_vector = dataset[target_attribute].to_numpy()
    index_best_attribute = best_attribute_to_split(dataset_matrix, classes_vector)
    best_attribute = attributes[index_best_attribute]

    tree = {str(best_attribute): {}}
    remaining_attributes = attributes.copy()
    remaining_attributes.remove(best_attribute)

    for value in numpy.unique(dataset[best_attribute]):
      sub_data = dataset.where(dataset[best_attribute] == value).dropna()
      subtree = id3(sub_data, remaining_attributes, target_attribute, parent_node_class)
      tree[best_attribute][value] = subtree

    return tree

def most_common_value(dataset, target_attribute):
  """
  Determina el valor mas comun del target_attribute en el conjunto de datos.

  Args:
      dataset (pandas.DataFrame): El conjunto de datos de entrada.
      target_attribute (str): El nombre de la columna que contiene las etiquetas o clases objetivo.

  Returns:
      El valor mas comun del target_attribute en el conjunto de datos.
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
  S = S.astype(int)
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

def train_model(model, dataset_train, target_train):
  """
  Entrena un modelo de machine learning utilizando un conjunto de datos de entrenamiento.

  Args:
      model: Un modelo de aprendizaje automático de scikit-learn (por ejemplo, RandomForestClassifier, DecisionTreeClassifier, etc.)
      dataset_train (pd.DataFrame o np.ndarray): El conjunto de características (atributos) para el entrenamiento.
      target_train (pd.Series o np.ndarray): El vector de etiquetas (target) correspondiente al conjunto de entrenamiento.
  """
  model.fit(dataset_train, target_train)

def evaluate_sklearn_model(model, dataset_test, target_test):
  """
  Evalúa el rendimiento de un modelo de machine learning utilizando un conjunto de datos de prueba.

  Args:
      model: Un modelo de aprendizaje automático entrenado de scikit-learn.
      dataset_test (pd.DataFrame o np.ndarray): El conjunto de características (atributos) para la evaluación.
      target_test (pd.Series o np.ndarray): El vector de etiquetas (target) correspondiente al conjunto de prueba.

  Returns:
      accuracy (float): La precisión del modelo en el conjunto de prueba.
      classification_report (str): Un informe detallado de clasificación que incluye métricas como precisión, recall y f1-score.
  """
  target_pred = model.predict(dataset_test)

  accuracy = metrics.accuracy_score(target_test, target_pred)
  report = metrics.classification_report(target_test, target_pred)

  print(f"Accuracy: {accuracy}")
  print("Classification Report:")
  print(report)

  return accuracy, report

def predict_id3(tree, sample):
  """
  Dado una muestra, predice el valor en cierto arbol generado por nuestro algoritmo id3
  (recordar que el arbol representa una funcion discreta, asi que lo que se esta haciendo
    es evaluar la funcion predecida en la muestra)

  Args:
      tree: Un arbol generado por id3
      sample: una muestra (por ejemplo una row del dataset)

  Returns:
      el valor del target attribute para esa muestra en el arbol tree
  """

  while isinstance(tree, dict):
    attribute = list(tree.keys())[0]
    attribute_value = sample[attribute]

    if attribute_value in tree[attribute]:
      tree = tree[attribute][attribute_value]
    else:
      return None

  return tree

def evaluate_id3_model(tree_id3, dataset_test, target_test, most_common_value):
  """
  Evalúa el rendimiento del modelo devuelto por nuestro algoritmo id3

  Args:
      tree_id3: El arbol devuelto por id3.
      dataset_test (pd.DataFrame o np.ndarray): El conjunto de características (atributos) para la evaluación.
      target_test (pd.Series o np.ndarray): El vector de etiquetas (target) correspondiente al conjunto de prueba.
      most_common_value: valor mas comun en el dataset para el target_attribute
  """

  predictions = [predict_id3(tree_id3, sample) for _, sample in dataset_test.iterrows()]
  predictions = [p if p is not None else most_common_value for p in predictions]
  target_test_for_id3 = target_test.astype(str)

  accuracy = metrics.accuracy_score(target_test_for_id3, predictions)
  # Ponemos el most common value para las samples no predecidas
  report = metrics.classification_report(target_test_for_id3, predictions, zero_division = most_common_value)

  print(f"Accuracy: {accuracy}")
  print("Classification Report:")
  print(report)

  return accuracy, report

####################################################################################################

# Cargar el archivo CSV
dataset = pandas.read_csv('lab1_dataset.csv')
dataset = dataset.drop('pidnum', axis='columns')

# Definir los atributos y el atributo objetivo
attributes = dataset.columns.values.tolist()
attributes.remove('cid')
target_attribute = 'cid'
mcv = most_common_value(dataset, target_attribute)

# Dividir los datos en conjuntos de entrenamiento y prueba
dataset_train, dataset_test, target_train, target_test = model_selection.train_test_split(dataset[attributes], dataset[target_attribute], test_size=0.2, random_state=42)

######## Construccion, entrenamiento y evaluacion de modelos #######

# ID3
print("ID3")
dataset_train_for_id3 = pandas.concat([dataset_train, target_train], axis=1)
tree_id3 = id3(dataset_train_for_id3, attributes, target_attribute)
evaluate_id3_model(tree_id3, dataset_test, target_test, mcv)

# DecisionTreeClassifier
print("DecisionTreeClassifier")
dtc_model = tree.DecisionTreeClassifier(random_state=42)
train_model(dtc_model, dataset_train, target_train)
_accuracy, _report = evaluate_sklearn_model(dtc_model, dataset_test, target_test)

# RandomForestClassifier
print("RandomForestClassifier")
rfc_model = ensemble.RandomForestClassifier(n_estimators=100, random_state=42)  # n_estimators es el número de árboles en el bosque
train_model(rfc_model, dataset_train, target_train)
_accuracy, _report = evaluate_sklearn_model(rfc_model, dataset_test, target_test)

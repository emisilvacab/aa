import pandas
import numpy
from sklearn import ensemble, model_selection, metrics, tree

ATTRIBUTES_REQUIRING_RANGES = ['time', 'age', 'wtkg', 'karnof', 'preanti', 'cd40', 'cd420', 'cd80', 'cd820']
TARGET_ATTRIBUTE = 'cid'

def id3(dataset, attributes, target_attribute, max_range_split = 0, parent_node_class = None):
  """
  Implementación del algoritmo ID3 para construir un árbol de decisión.
  La primer llamada a id3 debería ser: id3(dataset, attributes, target_attribute)
  La implementación soportada por id3 hace que el target_attribute solo pueda tener valor ser 1 o 0 en el dataset

  Args:
      dataset (pandas.DataFrame): El conjunto de datos de entrada, un DataFrame de pandas que contiene los datos.
      attributes (list): Lista de atributos disponibles para realizar divisiones (sin incluir a target_attribute).
      target_attribute (str): El nombre de la columna que contiene la etiqueta o clase objetivo.
      max_range_split (int): El número máximo de divisiones o rangos permitidos para los atributos numéricos.
      parent_node_class (str, opcional): La clase del nodo padre, utilizada en caso de que no queden más atributos.

  Returns:
      dict o str: Un árbol de decisión representado como un diccionario anidado, o un valor de clase.
  """

  if len(numpy.unique(dataset[target_attribute])) == 1:
    # Si todos los ejemplos tienen el mismo valor → etiquetar con ese valor
    return dataset[target_attribute]

  elif len(attributes) == 0:
    # Si no me quedan atributos → etiquetar con el valor más común
    return parent_node_class

  else:
    # parent_node_class es usado en caso de que no nos quedan mas atributos en la proxima llamada al id3
    parent_node_class = most_common_value(dataset, target_attribute)

    dataset_matrix = dataset[attributes].to_numpy()
    classes_vector = dataset[target_attribute].to_numpy()
    index_best_attribute = best_attribute_to_split(dataset_matrix, classes_vector)
    best_attribute = attributes[index_best_attribute]

    tree = {best_attribute: {}}
    remaining_attributes = attributes.copy()
    remaining_attributes.remove(best_attribute)

    if max_range_split != 0 and (best_attribute in ATTRIBUTES_REQUIRING_RANGES):
      cut_points = get_cut_points_based_on_target(dataset[best_attribute], dataset[target_attribute], max_range_split)
      ranges = numpy.concatenate(([float('-inf')], cut_points, [float('inf')]))

      for i in range(len(ranges) - 1):
        # Valores sub del dataset tal que ranges[i] > sub <= ranges[i+1]
        sub_data = dataset[(dataset[best_attribute] > ranges[i]) & (dataset[best_attribute] <= ranges[i + 1])]

        if sub_data.empty:
          subtree = parent_node_class
        else:
          subtree = id3(sub_data, remaining_attributes, target_attribute, max_range_split, parent_node_class)

        # (ranges[i], ranges[i+1]]
        tree[best_attribute][f"{ranges[i]} {ranges[i+1]}"] = subtree
    else:
      # Atributo no numérico, manejar normalmente
      for value in numpy.unique(dataset[best_attribute]):
        sub_data = dataset.where(dataset[best_attribute] == value).dropna()
        subtree = id3(sub_data, remaining_attributes, target_attribute, max_range_split, parent_node_class)
        tree[best_attribute][value] = subtree

    return tree

def get_cut_points_based_on_target(data, target, max_range_split):
  """
  Calcula los puntos de corte para un atributo numérico basados en cambios de clase en el valor objetivo.

  Args:
      data (pandas.Series o numpy.ndarray): El atributo numérico a dividir.
      target (pandas.Series o numpy.ndarray): El vector de etiquetas (valores objetivo).
      max_range_split (int): El número máximo de divisiones o rangos permitidos.

  Returns:
      list: Una lista de puntos de corte seleccionados.
  """
  # Convertir data y target a numpy arrays para evitar problemas de indexación con pandas
  data = numpy.array(data)
  target = numpy.array(target)

  # Ordenar los datos y los objetivos de acuerdo al atributo numérico
  sorted_indices = numpy.argsort(data)
  sorted_data = data[sorted_indices]
  sorted_target = target[sorted_indices]

  # Identificar los puntos donde cambia la clase
  cut_points = []
  for i in range(1, len(sorted_data)):
    if sorted_target[i] != sorted_target[i - 1]:
      cut_point = (sorted_data[i] + sorted_data[i - 1]) / 2.0  # Promedio de dos valores adyacentes
      cut_points.append(cut_point)

  # Si hay más puntos de corte que los permitidos, selecciona los más relevantes
  if len(cut_points) > max_range_split:
    cut_points = filter_cut_points_based_on_max_range_split(cut_points, max_range_split)

  return cut_points

def filter_cut_points_based_on_max_range_split(cut_points, max_range_split):
  """
    Selecciona puntos de corte basados en el número máximo de divisiones permitidas (max_range_split).
    (Únicamente para max_range_split= 2 o max_range_split=3)

    Args:
        cut_points (list o np.ndarray): Lista o array de puntos de corte calculados para dividir un atributo numérico.
        max_range_split (int): Número máximo de divisiones o rangos permitidos para el atributo numérico.

    Returns:
        list: Lista de puntos de corte seleccionados de acuerdo a max_range_split.
    """
  if max_range_split == 2:
    return [min(cut_points)]
  else:
    return [min(cut_points), max(cut_points)]

def most_common_value(dataset, target_attribute):
  """
  Determina el valor mas comun del target_attribute en el conjunto de datos.

  Args:
      dataset (pandas.DataFrame): El conjunto de datos de entrada.
      target_attribute (str): El nombre de la columna que contiene las etiquetas o clases objetivo.

  Returns:
      El valor mas comun del target_attribute en el conjunto de datos.
  """
  return dataset[target_attribute].mode()[0]

def set_range_to_value(value, ranges):
  for i in range(0, len(ranges) - 1):
    if value < ranges[i + 1]:
      return i

def add_ranges_for_attribute(dataset, attribute_name, max_range_split):
  cut_points = get_cut_points_based_on_target(dataset[attribute_name], dataset[TARGET_ATTRIBUTE], max_range_split)
  ranges = numpy.concatenate(([float('-inf')], cut_points, [float('inf')]))

  dataset[attribute_name] = dataset[attribute_name].apply(set_range_to_value, args=(ranges,))

def preprocessing_with_max_range_split(dataset, max_range_split):
  ans_dataset = dataset.copy()

  for attribute in ATTRIBUTES_REQUIRING_RANGES:
    add_ranges_for_attribute(ans_dataset, attribute, max_range_split)

  return ans_dataset

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

def train_sklearn_model(model, dataset_train, target_train):
  """
  Entrena un modelo de scikit-learn utilizando un conjunto de datos de entrenamiento.

  Args:
      model: Un modelo de aprendizaje automático de scikit-learn (por ejemplo, RandomForestClassifier, DecisionTreeClassifier, etc.)
      dataset_train (pandas.DataFrame o numpy.ndarray): El conjunto de características (atributos) para el entrenamiento.
      target_train (pandas.Series o numpy.ndarray): El vector de etiquetas (target) correspondiente al conjunto de entrenamiento.
  """
  model.fit(dataset_train, target_train)

def evaluate_sklearn_model(model, dataset_test, target_test):
  """
  Evalúa el rendimiento de un modelo de scikit-learn utilizando un conjunto de datos de prueba.

  Args:
      model: Un modelo de aprendizaje automático entrenado de scikit-learn.
      dataset_test (pandas.DataFrame o numpy.ndarray): El conjunto de características (atributos) para la evaluación.
      target_test (pandas.Series o numpy.ndarray): El vector de etiquetas (target) correspondiente al conjunto de prueba.

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

def predict_id3(tree, sample, default_value, use_ranges):
  """
  Dado una muestra, predice el valor en cierto arbol generado por nuestro algoritmo id3
  (recordar que el arbol representa una funcion discreta, asi que lo que se esta haciendo
    es evaluar la funcion predecida en la muestra)

  Args:
      tree: Un arbol generado por id3
      sample: Una muestra (por ejemplo una row del dataset)
      default_value: Usado para devolverlo en caso de que el tree no pueda predecir la sample.

  Returns:
      el valor del target attribute para esa muestra en el arbol tree
  """

  if isinstance(tree, dict):
    attribute = next(iter(tree))
    sample_value = sample[attribute]

    if attribute in ATTRIBUTES_REQUIRING_RANGES and use_ranges:
      for range_str in tree[attribute]:
        low, high = map(float, range_str.split())
        if low < sample_value <= high:
          subtree = tree[attribute][range_str]
          return predict_id3(subtree, sample, default_value, use_ranges)
    else:
      if sample_value in tree[attribute]:
        subtree = tree[attribute][sample_value]
        return predict_id3(subtree, sample, default_value, use_ranges)

    return default_value
  else:
    if isinstance(tree, pandas.Series):
      return tree.iloc[0]
    return tree

def evaluate_id3_model(tree_id3, dataset_test, target_test, most_common_value, use_ranges = True):
  """
  Evalúa el rendimiento del modelo devuelto por nuestro algoritmo id3

  Args:
      tree_id3: El arbol devuelto por id3.
      dataset_test (pandas.DataFrame o numpy.ndarray): El conjunto de características (atributos) para la evaluación.
      target_test (pandas.Series o numpy.ndarray): El vector de etiquetas (target) correspondiente al conjunto de prueba.
      most_common_value: valor mas comun en el dataset para el target_attribute
  """

  predictions = [predict_id3(tree_id3, sample, most_common_value, use_ranges) for _, sample in dataset_test.iterrows()]
  target_test_for_id3 = target_test.astype(int)

  accuracy = metrics.accuracy_score(target_test_for_id3, predictions)
  report = metrics.classification_report(target_test_for_id3, predictions)

  print(f"Accuracy: {accuracy}")
  print("Classification Report:")
  print(report)

  return accuracy, report
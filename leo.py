def select_attribute(dataset):
  """
  Calculate the better attribute

  Parameters:
  radius (float): The radius of the circle. Must be a non-negative number.

  Returns:
  string: The name of the attribute

  Raises:
  ValueError: If the radius is negative.

  Example:
  >>> calculate_area(5)
  78.53981633974483
  """



# Receives a boolean list, e.g: [1, 0, 1, 0]
# Returns its entropy: e.g: 1
def entropy(results):
  list_lenght = len(results)
  count_true = sum(results)
  count_false = list_lenght - count_true

  positive_proportion = count_true / list_lenght
  negative_proportion = count_false / list_lenght

  return - (positive_proportion * log(positive_proportion)) - (negative_proportion * log(negative_proportion))

def earning(dataset, attribute):


def possible_values(dataset, attribute):
  unique_values = set()

  with open('data.csv', mode='r') as file:
    reader = csv.DictReader(file)

    # Iterar sobre cada fila
    for row in reader:
      unique_values.add(row[attribute])

# Convertir el conjunto a una lista (si es necesario)
unique_values_list = list(unique_values)


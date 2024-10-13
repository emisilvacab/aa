import gymnasium as gym
from lab3 import *
import numpy as np
import matplotlib.pyplot as plt
import itertools

# Parámetros a buscar
gamma_values = [0.9, 0.95, 0.99]
epsilon_values = [1.0, 0.8, 0.5]
epsilon_min_values = [0.01, 0.05]
epsilon_decay_values = [0.995, 0.999]

# Crear todas las combinaciones de hiperparámetros
param_grid = list(itertools.product(gamma_values, epsilon_values, epsilon_min_values, epsilon_decay_values))

def ejecutar_episodio(agente, aprender = True, render = None):
  entorno = gym.make('LunarLander-v2', render_mode=render).env
  recompensa_total = 0
  termino = False
  truncado = False
  estado_anterior, info = entorno.reset()

  while not termino and not truncado:
      # Le pedimos al agente que elija entre las posibles acciones (0..entorno.action_space.n)
      accion = agente.elegir_accion(estado_anterior, entorno.action_space.n, aprender)

      # Realizamos la accion
      estado_siguiente, recompensa, termino, truncado, info = entorno.step(accion)

      if (aprender):
          agente.aprender(estado_anterior, estado_siguiente, accion, recompensa, termino)

      agente.fin_episodio()
      estado_anterior = estado_siguiente
      recompensa_total += recompensa
  entorno.close()
  return recompensa_total


# Función para entrenar el agente con un conjunto de parámetros
def entrenar_y_evaluar(agente_params, num_episodios=1000):
    gamma, epsilon, epsilon_min, epsilon_decay = agente_params

    # Inicializar el entorno y el agente con los parámetros actuales
    entorno = gym.make('LunarLander-v2').env
    bins_per_dim = 15
    NUM_BINS = [bins_per_dim, bins_per_dim, bins_per_dim, bins_per_dim, bins_per_dim, bins_per_dim, 2, 2]
    OBS_SPACE_HIGH = entorno.observation_space.high
    OBS_SPACE_LOW = entorno.observation_space.low
    OBS_SPACE_LOW[1] = 0
    bins = [
        np.linspace(OBS_SPACE_LOW[i], OBS_SPACE_HIGH[i], NUM_BINS[i] - 1)
        for i in range(len(NUM_BINS) - 2) # last two are binary
    ]
    entorno.close()

    agente = AgenteRL(bins, entorno.action_space.n, gamma=gamma, epsilon=epsilon, epsilon_min=epsilon_min, epsilon_decay=epsilon_decay)

    exitos = 0
    recompensa_episodios = []

    for i in range(num_episodios):
        recompensa = ejecutar_episodio(agente, aprender=True)

        if recompensa >= 200:
            exitos += 1

        recompensa_episodios.append(recompensa)

    tasa_exito = exitos / num_episodios
    recompensa_promedio = np.mean(recompensa_episodios)

    return tasa_exito, recompensa_promedio

# Entrenar y evaluar en todas las combinaciones de parámetros
resultados = []

for params in param_grid:
    print(f"Probando parámetros: gamma={params[0]}, epsilon={params[1]}, epsilon_min={params[2]}, epsilon_decay={params[3]}")
    tasa_exito, recompensa_promedio = entrenar_y_evaluar(params)
    resultados.append((params, tasa_exito, recompensa_promedio))

# Mostrar los mejores resultados
resultados.sort(key=lambda x: -x[1])  # Ordenar por tasa de éxito
for params, tasa_exito, recompensa_promedio in resultados[:5]:
    print(f"Parámetros: gamma={params[0]}, epsilon={params[1]}, epsilon_min={params[2]}, epsilon_decay={params[3]}, Tasa de éxito: {tasa_exito}, Recompensa promedio: {recompensa_promedio}")

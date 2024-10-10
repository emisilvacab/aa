import gymnasium as gym
from lab3 import *
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from pygame.locals import *

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


entorno = gym.make('LunarLander-v2').env
# Cuántos bins queremos por dimensión
bins_per_dim = 15

# Estado: (x, y, x_vel, y_vel, theta, theta_vel, pie_izq_en_contacto, pie_derecho_en_contacto)
NUM_BINS = [bins_per_dim, bins_per_dim, bins_per_dim, bins_per_dim, bins_per_dim, bins_per_dim, 2, 2]

#  Tomamos los rangos del entorno
OBS_SPACE_HIGH = entorno.observation_space.high
OBS_SPACE_LOW = entorno.observation_space.low
OBS_SPACE_LOW[1] = 0

# Los bins para cada dimensión
bins = [
    np.linspace(OBS_SPACE_LOW[i], OBS_SPACE_HIGH[i], NUM_BINS[i] - 1)
    for i in range(len(NUM_BINS) - 2) # last two are binary
]
entorno.close()

agente = AgenteRL(bins, entorno.action_space.n)
exitos = 0
recompensa_episodios = []
num_episodios = 1000
for i in range(num_episodios):
    recompensa = ejecutar_episodio(agente, aprender = True)

    # Los episodios se consideran exitosos si se obutvo 200 o más de recompensa total
    if (recompensa >= 200):
        exitos += 1

    recompensa_episodios += [recompensa]
print(f"Tasa de éxito APRENDIENDO: {exitos / num_episodios}. Se obtuvo {np.mean(recompensa_episodios)} de recompensa, en promedio")

for i in range(num_episodios):
    recompensa = ejecutar_episodio(agente, aprender = False)

    # Los episodios se consideran exitosos si se obutvo 200 o más de recompensa total
    if (recompensa >= 200):
        exitos += 1

    recompensa_episodios += [recompensa]
print(f"Tasa de éxito EXPLOTANDO: {exitos / num_episodios}. Se obtuvo {np.mean(recompensa_episodios)} de recompensa, en promedio")

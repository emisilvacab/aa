import gymnasium as gym
from lab3 import *
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from pygame.locals import *

def ejecutar_episodio(agente, aprender = True, render = None, max_iteraciones=500):
    entorno = gym.make('LunarLander-v2', render_mode=render).env

    iteraciones = 0
    recompensa_total = 0

    termino = False
    truncado = False
    estado_anterior, info = entorno.reset()
    while iteraciones < max_iteraciones and not termino and not truncado:
        # Le pedimos al agente que elija entre las posibles acciones (0..entorno.action_space.n)
        accion = agente.elegir_accion(estado_anterior, entorno.action_space.n, aprender)
        # Realizamos la accion
        estado_siguiente, recompensa, termino, truncado, info = entorno.step(accion)
        # Le informamos al agente para que aprenda
        if (aprender):
            agente.aprender(estado_anterior, estado_siguiente, accion, recompensa, termino)

        estado_anterior = estado_siguiente
        iteraciones += 1
        recompensa_total += recompensa
    if (aprender):
        agente.fin_episodio()
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
exitos_aprendiendo = 0
recompensa_episodios_aprendiendo = []
num_episodios_aprendiendo = 5000
for i in range(num_episodios_aprendiendo):
    recompensa = ejecutar_episodio(agente, aprender = True)

    # Los episodios se consideran exitosos si se obutvo 200 o más de recompensa total
    if (recompensa >= 200):
        exitos_aprendiendo += 1

    recompensa_episodios_aprendiendo += [recompensa]
print(f"Tasa de éxito APRENDIENDO: {exitos_aprendiendo / num_episodios_aprendiendo}. Se obtuvo {np.mean(recompensa_episodios_aprendiendo)} de recompensa, en promedio")

recompensa_episodios_explotando = []
num_episodios_explotando = 1000
exitos_explotando = 0

for i in range(num_episodios_explotando):
    recompensa = ejecutar_episodio(agente, aprender = False)

    # Los episodios se consideran exitosos si se obutvo 200 o más de recompensa total
    if (recompensa >= 200):
        exitos_explotando += 1

    recompensa_episodios_explotando += [recompensa]
print(f"Tasa de éxito EXPLOTANDO: {exitos_explotando / num_episodios_explotando}. Se obtuvo {np.mean(recompensa_episodios_explotando)} de recompensa, en promedio")

# Graficar recompensas para ambas fases
plt.plot(recompensa_episodios_aprendiendo, label='Aprendiendo')
plt.plot(recompensa_episodios_explotando, label='Explotando')
plt.xlabel('Episodio')
plt.ylabel('Recompensa Total')
plt.title('Comparación de recompensas durante aprendizaje y explotación')
plt.legend()
plt.show()

# Condición: cayendo rápido y sin rotación
bins_cayendo_rapido = list(range(0, 7))  # Velocidad vertical rapida (negativa)
bins_sin_rotacion = list(range(5, 10))  # Ángulo y velocidad angular cerca de 0
contador_acciones = [0, 0, 0, 0]
for estado, valores_q in agente.q_table.items():
    _, _, x_vel_bin, y_vel_bin, theta_bin, theta_vel_bin, pie_izq, pie_der = estado
    if (y_vel_bin in bins_cayendo_rapido and theta_bin in bins_sin_rotacion and theta_vel_bin in bins_sin_rotacion):
        accion_recomendada = np.argmax(valores_q)
        contador_acciones[accion_recomendada] += 1
print('Condición: cayendo rápido y sin rotación')
print(f'Acción 0 (no hacer nada): {contador_acciones[0]} veces')
print(f'Acción 1 (activar propulsor izquierdo): {contador_acciones[1]} veces')
print(f'Acción 2 (activar motor principal): {contador_acciones[2]} veces')
print(f'Acción 3 (activar propulsor derecho): {contador_acciones[3]} veces')

# Condición: rotación sin control hacia la izquierda
bins_rotacion_izquierda = list(range(0, 5))  # Theta_vel negativa (sentido antihorario)
bins_inclinacion_izquierda = list(range(0, 5))  # Theta negativo (inclinación hacia la izquierda)
contador_acciones = [0, 0, 0, 0]
for estado, valores_q in agente.q_table.items():
    _, _, _, _, theta_bin, theta_vel_bin, pie_izq, pie_der = estado
    if (theta_vel_bin in bins_rotacion_izquierda and theta_bin in bins_inclinacion_izquierda):
        accion_recomendada = np.argmax(valores_q)
        contador_acciones[accion_recomendada] += 1
print('Condición: rotación sin control hacia la izquierda')
print(f'Acción 0 (no hacer nada): {contador_acciones[0]} veces')
print(f'Acción 1 (activar propulsor izquierdo): {contador_acciones[1]} veces')
print(f'Acción 2 (activar motor principal): {contador_acciones[2]} veces')
print(f'Acción 3 (activar propulsor derecho): {contador_acciones[3]} veces')

# Condición: rotación sin control hacia la derecha
bins_rotacion_derecha = list(range(10, 15))  # Theta_vel positiva (sentido horario)
bins_inclinacion_derecha = list(range(10, 15))  # Theta positiva (inclinación hacia la derecha)
contador_acciones = [0, 0, 0, 0]
for estado, valores_q in agente.q_table.items():
    _, _, _, _, theta_bin, theta_vel_bin, pie_izq, pie_der = estado
    if (theta_vel_bin in bins_rotacion_derecha and theta_bin in bins_inclinacion_derecha):
        accion_recomendada = np.argmax(valores_q)
        contador_acciones[accion_recomendada] += 1
print('Condición: rotación sin control hacia la derecha')
print(f'Acción 0 (no hacer nada): {contador_acciones[0]} veces')
print(f'Acción 1 (activar propulsor izquierdo): {contador_acciones[1]} veces')
print(f'Acción 2 (activar motor principal): {contador_acciones[2]} veces')
print(f'Acción 3 (activar propulsor derecho): {contador_acciones[3]} veces')



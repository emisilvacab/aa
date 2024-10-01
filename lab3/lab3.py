# class Agente:
#     def elegir_accion(self, estado, max_accion, explorar = True) -> int:
#         """Elegir la accion a tomar en el estado actual y el espacio de acciones
#             - estado_anterior: el estado desde que se empezó
#             - estado_siguiente: el estado al que se llegó
#             - accion: la acción que llevo al agente desde estado_anterior a estado_siguiente
#             - recompensa: la recompensa recibida en la transicion
#             - terminado: si el episodio terminó
#         """
#         pass

#     def aprender(self, estado_anterior, estado_siguiente, accion, recompensa, terminado):
#         """Aprender a partir de la tupla
#             - estado_anterior: el estado desde que se empezó
#             - estado_siguiente: el estado al que se llegó
#             - accion: la acción que llevo al agente desde estado_anterior a estado_siguiente
#             - recompensa: la recompensa recibida en la transicion
#             - terminado: si el episodio terminó en esta transición
#         """
#         pass


# import random

# class AgenteAleatorio(Agente):
#     def elegir_accion(self, estado, max_accion, explorar = True) -> int:
#         # Elige una acción al azar
#         return random.randrange(max_accion)

#     def aprender(self, estado_anterior, estado_siguiente, accion, recompensa, terminado):
#         # No aprende
#         pass


# def ejecutar_episodio(agente, aprender = True, render = None):
#   entorno = gym.make('LunarLander-v2', render_mode=render).env

#   iteraciones = 0
#   recompensa_total = 0

#   termino = False
#   truncado = False
#   estado_anterior, info = entorno.reset()
#   while not termino and not truncado:
#       # Le pedimos al agente que elija entre las posibles acciones (0..entorno.action_space.n)
#       # Si no estamos aprendiendo, explotamos sin explorar
#       accion = agente.elegir_accion(estado_anterior, entorno.action_space.n, not aprender)
#       # Realizamos la accion
#       estado_siguiente, recompensa, termino, truncado, info = entorno.step(accion)
#       # Le informamos al agente para que aprenda
#       if (aprender):
#           agente.aprender(estado_anterior, estado_siguiente, accion, recompensa, termino)

#       estado_anterior = estado_siguiente
#       iteraciones += 1
#       recompensa_total += recompensa
#   env.close()
#   return recompensa_total


# # Nota: hay que transformar esta celda en código para ejecutar (Esc + y)

# # Ejecutamos un episodio con el agente aleatorio y modo render 'human', para poder verlo
# ejecutar_episodio(AgenteAleatorio(), render = 'human')


# agente = AgenteAleatorio()
# recompensa_episodios = []

# exitos = 0
# num_episodios = 100
# for i in range(num_episodios):
#     recompensa = ejecutar_episodio(agente)
#     # Los episodios se consideran exitosos si se obutvo 200 o más de recompensa total
#     if (recompensa >= 200):
#         exitos += 1
#     recompensa_episodios += [recompensa]

# import numpy
# print(f"Tasa de éxito: {exitos / num_episodios}. Se obtuvo {numpy.mean(recompensa_episodios)} de recompensa, en promedio")


# class AgenteRL(Agente):
#     # Agregar código aqui

#     # Pueden agregar parámetros al constructor
#     def __init__(self) -> None:
#         super().__init__()
#         # Agregar código aqui

#     def elegir_accion(self, estado, max_accion, explorar = True) -> int:
#         # Agregar código aqui
#         return 0

#     def aprender(self, estado_anterior, estado_siguiente, accion, recompensa, terminado):
#         # Agregar código aqui
#         pass

#     def fin_episodio(self):
#         # Agregar código aqui
#         pass


# # Nota: hay que transformar esta celda en código para ejecutar (Esc + y)
# # Advertencia: este bloque es un loop infinito si el agente se deja sin implementar

# entorno = gym.make('LunarLander-v2').env
# agente = AgenteRL()
# exitos = 0
# recompensa_episodios = []
# num_episodios = 100
# for i in range(num_episodios):
#     recompensa = ejecutar_episodio(agente, aprender = False)
#     # Los episodios se consideran exitosos si se obutvo 200 o más de recompensa total
#     if (recompensa >= 200):
#         exitos += 1
#     recompensa_episodios += [recompensa]
# print(f"Tasa de éxito: {exitos / num_episodios}. Se obtuvo {numpy.mean(recompensa_episodios)} de recompensa, en promedio")
import random
import numpy as np
import gymnasium as gym
class Agente:
    def elegir_accion(self, estado, max_accion, explorar = True) -> int:
        """Elegir la accion a tomar en el estado actual y el espacio de acciones
            - estado_anterior: el estado desde que se empezó
            - estado_siguiente: el estado al que se llegó
            - accion: la acción que llevo al agente desde estado_anterior a estado_siguiente
            - recompensa: la recompensa recibida en la transicion
            - terminado: si el episodio terminó
        """
        pass

    def aprender(self, estado_anterior, estado_siguiente, accion, recompensa, terminado):
        """Aprender a partir de la tupla
            - estado_anterior: el estado desde que se empezó
            - estado_siguiente: el estado al que se llegó
            - accion: la acción que llevo al agente desde estado_anterior a estado_siguiente
            - recompensa: la recompensa recibida en la transicion
            - terminado: si el episodio terminó en esta transición
        """
        pass
class AgenteRL(Agente):
    def __init__(self, bins, gamma=0.9, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995) -> None:
        super().__init__()
        self.gamma = gamma
        self.epsilon = epsilon  # Factor de exploración
        self.epsilon_min = epsilon_min  # Mínimo valor de epsilon para la política epsilon-greedy
        self.epsilon_decay = epsilon_decay  # Decaimiento de epsilon para disminuir exploración
        self.bins = bins
        self.q_table = {}  # Tabla Q para almacenar los valores Q de los estados y acciones

    def _discretize_state(self, state):
        """Discretiza el estado continuo en una tupla de indices discretos"""

        state_disc = list()
        for i in range(len(state)):
            if i >= len(self.bins):
                state_disc.append(int(state[i]))
            else:
                state_disc.append(np.digitize(state[i], self.bins[i]))
        return tuple(state_disc)

    def elegir_accion(self, estado, max_accion, explorar=True) -> int:
        """Elige una acción usando la política epsilon-greedy."""

        estado = self._discretize_state(estado)

        if explorar and np.random.rand() < self.epsilon:  # Política epsilon-greedy
            return random.randint(0, max_accion - 1)  # Acción aleatoria
        else:
            if estado not in self.q_table:
                self.q_table[estado] = np.zeros(max_accion)  # Inicializa con ceros si no existe
            return np.argmax(self.q_table[estado])  # Acción óptima

    def aprender(self, estado_anterior, estado_siguiente, accion, recompensa, terminado):
        """Actualiza la tabla Q usando la fórmula de Q-learning."""

        estado_anterior = self._discretize_state(estado_anterior)
        estado_siguiente = self._discretize_state(estado_siguiente)
        q_valor_actual = self.q_table[estado_anterior][accion]

        max_q_siguiente = np.max(self.q_table[estado_siguiente]) if estado_siguiente in self.q_table else 0
        q_valor_actual = recompensa + self.gamma * max_q_siguiente * (1 - terminado)
        self.q_table[estado_anterior][accion] = q_valor_actual

    def fin_episodio(self):
        """Reduce epsilon para disminuir la exploración en episodios futuros."""

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

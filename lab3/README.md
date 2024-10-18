# MAC
brew install swig
# LINUX
sudo apt-get install swig3.0
sudo apt-get install python3.10-dev

pip install pygame
pip install gymnasium
pip install 'gymnasium[box2d]'

##### Esto de abajo no me anduvo

  !pip3 install cmake gymnasium scipy numpy gymnasium[box2d] pygame==2.6.0 swig

Tal vez tengan que ejecutar lo siguiente en sus máquinas (ubuntu 20.04)
  sudo apt-get remove swig
  sudo apt-get install swig3.0
  sudo ln -s /usr/bin/swig3.0 /usr/bin/swig

En windows tambien puede ser necesario MSVC++



Analizar los resultados de la ejecución anterior, incluyendo:
 * Un análisis de los parámetros utilizados en el algoritmo (aprendizaje, política de exploración)
 * Un análisis de algunos 'cortes' de la matriz Q y la política (p.e. qué hace la nave cuando está cayendo rápidamente hacia abajo, sin rotación)
 * Un análisis de la evolución de la recompensa promedio
 * Un análisis de los casos de éxito
 * Un análisis de los casos en el que el agente falla
 * Qué limitante del agente de RL les parece que afecta más negativamente su desempeño.
	- implementar lo de ir para atrás que habla en el teórico para mejorar los tiempos
	- usamos la fórmula no determinista pero con alfa fijo, se podría intentar con el alfa variable

  ¿Los fallos son más frecuentes al inicio del entrenamiento (indicando que el agente estaba aprendiendo) o se mantienen constantes hasta el final?

Si los fallos siguen ocurriendo hacia el final, ¿hay estados particulares que siempre llevan al fracaso?

Examina qué decisiones tomó el agente en esos episodios fallidos. Puede ser que ciertos estados críticos o transiciones estén llevando sistemáticamente a un mal desempeño.

hay que ver cuantos hay con < 200 y en que episodios pasaron


¿Cuántos episodios exitosos tuviste (aquellos con recompensa >= 200)?

¿Hay una concentración de recompensas en un rango muy alto (indicativo de un comportamiento muy optimizado) o la mayoría de los éxitos son marginales (justo por encima de 200)?

¿La cantidad de éxitos aumenta hacia el final de los episodios o es constante?


hay que ver cuantos hay con >= 200 y en que episodios pasaron


¿La curva muestra una mejora constante o hay fluctuaciones considerables? Un aumento gradual es indicativo de que el agente está aprendiendo.

¿Hay estancamiento o decrecimiento en la recompensa promedio? Esto podría indicar que el agente llegó a un óptimo local o que la política de exploración ya no es efectiva.

¿Hubo algún cambio brusco en la curva? Si es así, intenta correlacionar esto con algún cambio en el comportamiento del agente, como cambios en la política de exploración o el aprendizaje.
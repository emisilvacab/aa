notebook https://colab.research.google.com/drive/1oizUdeYxSKr5zJ3loWJU70x1rPigF1JZ#scrollTo=lhPGTKc5Cu-h

https://eva.fing.edu.uy/mod/forum/discuss.php?d=304427
hola:

se encuentra disponible la letra de la tarea 1. algunas consideraciones generales sobre las tarea y sobre la tarea 1 en particular:

1. Sobre las tareas en general

las primeras tareas son ejercicios de implementación de algoritmos de aprendizaje;  es muy importante el reporte de las pruebas realizadas, los resultados y su análisis.

1.1 implementación

la implementación se debe realizar en la última versión estable de python (3.12.x)  y no se pueden utilizar bibliotecas auxiliares que implementen lo que se pide resolver por tarea. por ejemplo, no vale usar el algoritmo X de scikit-learn para resolver un problema que pide implementar, precisamente, ese algoritmo X. sí se pueden utilizar bibliotecas auxiliares para el manejo de datos, por ejemplo, pandas para cargar un csv en una tabla. 

ante la duda de si utilizar algo o no, por favor, pregunten en el foro. 

1.2 informe

junto con la implementación pedida, deben entregar un pequeño informe con el modelado, las pruebas realizadas, los resultados obtenidos, etc. el informe a entregar debe ser un jupyter notebook;  pueden encontrar uno de referencia en el repositorio del curso. ustedes, por supuesto, pueden modificarlo según su criterio; eso sí, por favor, eviten repetir el teórico en los informes.

1.3 entrega
es obligatorio entregar los trabajos en fecha. sin embargo, cada grupo cuenta con cuatro días de prórroga que pueden utilizar cuando quieran, juntos o separados, a lo largo de todo el curso; la única condición es que nos avisen el día del vencimiento que van a utilizar uno de esos días. 

el uso de días de prórroga no altera la calificación de la tarea; entregar antes de plazo, tampoco. ni a favor, ni en contra.

los trabajos se entregan únicamente por EVA, ya sea en fecha, ya sea luego de pedir prórrogas. no entreguen, bajo ningún concepto, sus trabajos por correo electrónico.

1.4.- sobre las entregas

i) el informe no puede tener faltas de ortografía.  (taza por tasa, osea/ósea por 'o sea', etc.)

ii) el informe que les pedimos no es un reporte de programación, sino más bien que el enfoque es sobre la experimentación: 

(a) la decisiones de modelado  que tomaron: cómo preprocesaron los datos, qué cosas tuvieron en cuenta a la hora de modificar el algoritmo, qué hiperparámetros probaron en los algoritmos ya implementados, etc.

(b) la presentación de los resultados: ¿qué error obtuvieron?, ¿cómo varían los resultados?, etc. 

iii) los  resultados deben estar resumidos, por ejemplo, en una tabla. esto es importante para poder comparar las distintas variantes implementadas y los tests en distintos escenarios.

 iv)  aunque esto no es un curso de programación, el código es parte de la entrega. por favor, sean prolijos y agreguen comentarios a su código. 

1.5.- dudas

las dudas de letra, alcance, librerías a utilizar, etc. háganlas, por favor, en el foro.

les recuerden que:

"Las dudas o consultas pueden ser publicada en el Foro, siempre que no implique violar las condiciones de trabajo individual establecidas por el InCo. En particular, está terminantemente prohibido publicar la resolución de (o parte de) un ejercicio entregable; en caso que esto suceda, todos los integrantes del grupo perderán automáticamente el curso."

2. Tarea 1

como primera tarea se les pide implementar el algoritmo ID3 básico, con procesamiento de datos numéricos en rangos y con un hiperparámetro que permite determinar cuál es número máximo de rangos en que se puede, precisamente, partir al atributo numérico

se les piden distintas pruebas de entrenamiento: con y sin preprocesamiento de datos, con algoritmos ya implementados... y deben comparar todos los resultados. 

aunque lo pueden bajar del vínculo dado en la letra, les dejamos en el repositorio los archivos con los datos y la descripción de atributos.

el algoritmo ID3 lo vamos a ver hoy en clase y la parte de preprocesamiento y evaluación quedará completa la semana próxima cuando veamos metodología.

la fecha límite de entrega de esta tarea es el miércoles 4 de setiembre (inclusive).

saludos,
d.-

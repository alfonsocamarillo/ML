'''
Modelos no supervisados

Clustering:
Aunque similar al anterior puesto que trata de clasificar los datos en grupos específicos, 
al tratarse de un algoritmo no supervisado, no precisa de información previamente etiquetada. 

Contesta a cuestiones tales como 
¿segmentación de clientes en base a su gasto?,
¿agrupación de productos?, etc.

'''
import os
from numpy import unique
from numpy import where
import pandas as pd
from sklearn.cluster import Birch
from matplotlib import pyplot

os.chdir('D:\\Cursos\\GEM\\Diplomados\\Especializacion en estadistica y cincia de datos\\Modulo 4 Programacion con python\\ML')
df = pd.read_csv(r'costs-purchases.csv')

f1 = df['Spend'].values
f2 = df['Purchases'].values
features = list(zip(f1, f2))

model = Birch(threshold=0.01, n_clusters=3)
model.fit(features)

yhat = model.predict(features)
clusters = unique(yhat)
for cluster in clusters:
    row_ix = where(yhat == cluster)
    pyplot.scatter(f1[row_ix], f2[row_ix])

pyplot.show()

'''
Dimensionality reduction
Se centra en la reducción de variables con el objetivo de organizar los datos según patrones 
más simples en busca de las características esenciales. 
Como modelo no supervisado que es, se desconoce a priori cómo están etiquetados los datos.

Para interpretar un gráfico en un modelo no supervisado de clustering en Python, 
como el algoritmo de K-Means, podemos seguir estos pasos:

+++Puntos de datos: Al igual que en el caso de clasificación supervisada, los puntos en el 
gráfico representan las instancias de datos en nuestro conjunto de datos. Cada punto no tiene
una etiqueta de clase específica, ya que el clustering es un enfoque no supervisado y no se 
requieren etiquetas previas para agrupar los datos.

+++Centroides: En un gráfico de clustering K-Means, también veremos los centroides, que son los 
puntos que representan los centros de los clusters encontrados por el algoritmo. Estos centroides 
suelen estar marcados con un símbolo diferente o un color para distinguirlos de los puntos de datos.

+++Clusters: Los clusters son los grupos de puntos de datos que se han agrupado juntos debido a su 
similitud.En el gráfico, estos clusters pueden estar delineados por contornos o áreas de color para 
visualizar mejor la agrupación.

+++Número de clusters: Una parte importante de la interpretación es determinar el número óptimo de 
clusters. Esto puede lograrse observando cómo cambia la agrupación de los datos a medida que se aumenta
o disminuye el número de clusters. A menudo, se utiliza un método como el codo (elbow method) o el 
coeficiente de silueta para determinar el número óptimo de clusters.

+++Distribución de los datos: Observar cómo se distribuyen los puntos de datos dentro de cada cluster puede 
proporcionar información sobre la estructura de los datos y cómo están relacionados.

+++Interpretación de la separación entre clusters: También es importante analizar la separación 
entre los clusters. ¿Están claramente separados o se superponen? Esto puede indicar la calidad del 
clustering y la interpretabilidad de los grupos encontrados.

+++Análisis de los centroides: Examinar los centroides puede proporcionar información sobre las 
características medias de cada cluster y ayudar a interpretar qué representa cada cluster.

En resumen, al interpretar un gráfico en un modelo no supervisado de clustering en Python, 
es importante observar la distribución de los datos, la separación entre clusters, 
el número de clusters y los centroides para comprender mejor la estructura de los datos y 
la agrupación resultante.
'''
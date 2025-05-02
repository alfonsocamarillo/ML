'''
Modelos supervisados

Classification:
Este algoritmo supervisado se centra en agrupar valores categóricos. 

Por ejemplo, 
¿están nuestros clientes satisfechos?, 
¿será devuelto el producto?, etc. 
Es decir, cuantifica la tipología de los datos. 
A distinción del anterior, donde el resultado es contínuo, los valores son discretos. 
Gracias a lo cual se puede realizar la segmentación buscada. 
En el siguiente ejemplo práctico se ha desarrollado un algoritmo KNN 
en el que además se ha precalculado la mejor aproximación y el número K óptimo. 
Este número representa la cantidad de vecinos a agrupar entorno a valores concretos.

Nota:El algoritmo K-Nearest Neighbors (KNN) es un método de aprendizaje supervisado 
utilizado principalmente para problemas de clasificación y, en algunos casos, para regresión.
'''

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV

os.chdir('D:\\Cursos\\GEM\\Diplomados\\Especializacion en estadistica y cincia de datos\\Modulo 4 Programacion con python\\ML\\')
df = pd.read_csv(r'costs-purchases.csv')

f1 = df['Spend']
f2 = df['Purchases']
features = list(zip(f1, f2))
le = preprocessing.LabelEncoder()
label = le.fit_transform(df["SourceMedium"])
h = 100

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#00AAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#3341FF'])
k_range = list(range(1, 100))
weight_options = ["uniform", "distance"]
param_grid = dict(n_neighbors=k_range, weights=weight_options)
knn = KNeighborsClassifier()
knn = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy')
knn.fit(features, label)

model = KNeighborsClassifier(n_neighbors=knn.best_params_['n_neighbors'], weights=knn.best_params_['weights'])
model.fit(features, label)

x_min, x_max = f1.min() - 1, f1.max() + 1
y_min, y_max = f2.min() - 1, f2.max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
predicted = model.predict(np.c_[xx.ravel(), yy.ravel()])
predicted = predicted.reshape(xx.shape)

plt.figure()
plt.pcolormesh(xx, yy, predicted, cmap=cmap_light)
plt.scatter(f1, f2, c=label, cmap=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("KNN classification (k = %i)" % (knn.best_params_['n_neighbors']))
plt.show()

'''
Además de todo esto, existen otros algoritmos como pueden ser los Árboles de decisión 
que pueden hacer uso de ambas técnicas: regresión y clasificación. 
De hecho, este tipo de implementación muchas veces se usa como complemento 
para validar otros modelos. 

Ejemplo de interpretación de un grafico KNN classification (k = 12)
Para interpretar un gráfico de clasificación KNN con un valor de "k" igual a 12, 
primero debemos entender cómo se visualiza la clasificación en un espacio bidimensional 
(o tridimensional si tenemos tres características). 
Supongamos que estamos trabajando en un problema de clasificación binaria, 
donde tenemos dos clases: Clase A y Clase B.

Aquí hay un ejemplo de cómo podríamos interpretar un gráfico de clasificación 
KNN con un valor de "k" igual a 12:

+++Puntos de datos: En el gráfico, cada punto representa una instancia de datos 
en nuestro conjunto de datos. Cada punto puede estar marcado con un color o un 
símbolo diferente según su clase real (por ejemplo, puntos azules para la Clase A 
y puntos rojos para la Clase B).

+++Fronteras de decisión: Las fronteras de decisión son las líneas, curvas o regiones 
que separan las diferentes clases en el espacio de características. Estas fronteras se 
derivan de los "k" vecinos más cercanos para cada punto del espacio de características.

+++Fronteras suaves vs. Fronteras nítidas: Las fronteras de decisión en un gráfico KNN 
pueden ser suaves o nítidas dependiendo de la complejidad del problema y del valor de "k". 
Con un valor de "k" mayor, las fronteras tienden a ser más suaves, lo que puede conducir a un
modelo más generalizado, mientras que con un valor de "k" menor, las fronteras tienden a ser más nítidas,
lo que puede llevar a un sobreajuste.

+++Interpretación del valor de "k": Un valor de "k" igual a 12 significa que para cada punto en
el espacio de características, el algoritmo KNN considera las 12 instancias de datos más cercanas 
para tomar una decisión de clasificación. Por lo tanto, las fronteras de decisión en el gráfico se
derivan de la contribución de estas 12 instancias más cercanas.

+++Clasificación de nuevos puntos: Una vez que el modelo está entrenado y las fronteras de decisión 
están definidas, podemos clasificar nuevos puntos en el espacio de características según la región a
la que pertenezcan. Esto se hace considerando los "k" vecinos más cercanos y aplicando la regla de 
votación para determinar la clase más probable del nuevo punto.

En resumen, un gráfico de clasificación KNN con un valor de "k" igual a 12 nos permite visualizar 
cómo se separan las diferentes clases en el espacio de características y cómo se clasificarían los 
nuevos puntos según la regla de los "k" vecinos más cercanos.

'''
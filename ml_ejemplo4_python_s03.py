'''
Modelos no supervisados

PCA (Principal Component Analysis):  es una técnica de reducción de dimensionalidad ampliamente 
utilizada en el análisis de datos y el aprendizaje automático. Su objetivo es transformar un 
conjunto de datos de alta dimensionalidad en un conjunto de dimensiones más bajas 
(llamadas componentes principales) mientras se conserva la mayor cantidad posible de la
variabilidad original de los datos.

El proceso de PCA implica los siguientes pasos:
+++Centrar los datos: El primer paso es centrar los datos, lo que implica restar la media 
de cada característica de los datos. Esto asegura que el origen de nuestro nuevo espacio de
características esté en el centro de los datos.

+++Calcular la matriz de covarianza: Luego, se calcula la matriz de covarianza de los datos
centrados. La covarianza es una medida de cómo dos variables varían juntas.

+++Calcular los autovalores y autovectores: A continuación, se calculan los autovalores y 
autovectores de la matriz de covarianza. Los autovectores son los vectores propios que definen
nuevas direcciones en el espacio de características, y los autovalores representan la magnitud 
de la varianza explicada por cada autovector.

+++Seleccionar componentes principales: Se ordenan los autovectores según sus autovalores 
correspondientes en orden descendente. Los autovectores con los autovalores más grandes (mayor
varianza)se convierten en los componentes principales.

+++Proyectar los datos en el nuevo espacio: Finalmente, se proyectan los datos originales en el
nuevo espacio de características definido por los componentes principales seleccionados.

Las aplicaciones de PCA incluyen la reducción de la dimensionalidad para la visualización de datos, 
la eliminación de la multicolinealidad en conjuntos de datos de alta dimensión, 
la compresión de datos para su almacenamiento eficiente y la extracción de características 
para mejorar el rendimiento de los modelos de aprendizaje automático.

La interpretación de los componentes principales suele implicar analizar los
autovectores para determinar qué características originales contribuyen más a cada componente principal, 
lo que puede proporcionar información sobre la estructura subyacente de los datos.

'''

import os
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


os.chdir('D:\\Cursos\\GEM\\Diplomados\\Especializacion en estadistica y cincia de datos\\Modulo 4 Programacion con python\\ML\\')
df = pd.read_csv(r'costs-purchases.csv')

features = ['Spend', 'Purchases'] 
x = df.loc[:, features].values 
y = df.loc[:, ['SourceMedium']].values 

x = StandardScaler().fit_transform(x) 
pca = PCA(n_components=2) 
principalComponents = pca.fit_transform(x) 
principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2']) 
finalDf = pd.concat([principalDf, df['SourceMedium']], axis = 1) 

fig = plt.figure(figsize = (8,8)) 
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('PC1', fontsize = 15) 
ax.set_ylabel('PC2', fontsize = 15) 
ax.set_title('PCA Analysis', fontsize = 20) 
targets = ['facebookAds', 'googleAds', 'twitterAds'] 
colors = ['r', 'g', 'b'] 
for target, color in zip(targets, colors):     
    indicesToKeep = finalDf['SourceMedium'] == target     
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1'], finalDf.loc[indicesToKeep, 'principal component 2'], c=color, s=50) 
ax.legend(targets) 
ax.grid() 

plt.show() 
print(pca.explained_variance_ratio_)

'''
Cuando se trata de interpretar un gráfico generado por un Análisis de Componentes Principales (PCA),
es fundamental comprender qué representa PCA y cómo se utiliza para reducir la dimensionalidad de los 
datos mientras se conserva la mayor cantidad posible de su variabilidad.

PCA es una técnica de reducción de dimensionalidad que transforma un conjunto de datos de alta
dimensionalidad en un conjunto de dimensiones más bajas (llamadas componentes principales) que representan 
la mayor cantidad de variabilidad en los datos originales. Estas nuevas dimensiones son combinaciones 
lineales de las variables originales.

Aquí hay algunas pautas para interpretar un gráfico generado por PCA:

+++Componentes Principales (PC): En un gráfico de PCA, cada eje representa un componente principal.
Por lo general, los componentes principales están ordenados en función de la cantidad de variabilidad 
que explican en los datos. El primer componente principal captura la mayor cantidad de variabilidad,
el segundo componente principal captura la siguiente mayor cantidad de variabilidad y así sucesivamente.

+++Varianza explicada: Es importante verificar cuánta varianza explica cada componente principal. 
Esto se puede ver en un gráfico de barras o en un gráfico de línea que muestra la proporción de varianza 
explicada por cada componente. Por lo general, se busca un punto de inflexión en el gráfico donde la cantidad 
de varianza explicada comienza a disminuir rápidamente.

+++Gráfico de dispersión de componentes principales: Un gráfico de dispersión de componentes principales muestra 
la representación de los datos en el espacio de los componentes principales. Cada punto en el gráfico representa 
una observación del conjunto de datos, y su posición está determinada por sus valores en los componentes principales.
Esto permite visualizar la estructura y la agrupación de los datos en función de sus componentes principales.

+++Interpretación de la carga de las variables: También es útil examinar cómo se relacionan las variables originales 
con los componentes principales. Esto se puede hacer mediante la carga de las variables en cada componente principal.
Una carga alta indica una fuerte contribución de la variable al componente principal. Esto puede ayudar a comprender qué 
características o variables están contribuyendo más a la variabilidad observada en los datos.

+++Interpretación de la separación de grupos: Si los datos tienen grupos conocidos o clases, es importante observar 
cómo se separan en el espacio de los componentes principales. Una buena separación entre grupos en el espacio de los
componentes principales puede indicar que estos grupos son fácilmente distinguibles incluso después de la reducción de
la dimensionalidad.

En resumen, al interpretar un gráfico generado por PCA, es esencial comprender cómo los componentes principales 
representan la variabilidad en los datos originales y cómo las observaciones se distribuyen en el espacio de los
componentes principales. Esto puede proporcionar información valiosa sobre la estructura y las relaciones subyacentes 
en los datos.
'''
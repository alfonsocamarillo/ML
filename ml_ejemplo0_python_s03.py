'''
Scikit-learn

Python es uno de los lenguajes de programación que domina dentro del ámbito de la estadística, 
data mining y machine learning. Al tratarse de un software libre, innumerables usuarios han
podido implementar sus algoritmos, dando lugar a un número muy elevado de librerías donde
encontrar prácticamente todas las técnicas de machine learning existentes. 

Sin embargo, esto tiene un lado negativo, cada paquete tiene una sintaxis,estructura e 
implementación propia, lo que dificulta su aprendizaje. Scikit-learn, es una librería de código
abierto que unifica bajo un único marco los principales algoritmos y funciones, facilitando en 
gran medida todas las etapas de preprocesado, entrenamiento, optimización y validación de
modelos predictivos.

Etapas de un problema de machine learning:

+++Definir el problema: 
¿Qué se pretende predecir? 
¿De qué datos se dispone? o 
¿Qué datos es necesario conseguir?

+++Explorar y entender los datos que se 
van a emplear para crear el modelo.

+++Métrica de éxito: 
definir una forma apropiada de cuantificar 
cómo de buenos son los resultados obtenidos.

+++Preparar la estrategia para evaluar el modelo: 
separar las observaciones en un conjunto de entrenamiento,
un conjunto de validación (o validación cruzada) y un conjunto de test.
Es muy importante asegurar que ninguna información del conjunto de test
participa en el proceso de entrenamiento del modelo.

+++Preprocesar los datos:
aplicar las transformaciones necesarias para que los datos puedan ser
interpretados por el algoritmo de machine learning seleccionado.

+++Ajustar un primer modelo capaz de superar unos resultados mínimos. 
Por ejemplo, en problemas de clasificación, el mínimo a superar es el
porcentaje de la clase mayoritaria (la moda). En un modelo de regresión,
la media de la variable respuesta.

+++Gradualmente, mejorar el modelo incorporando-creando nuevas variables 
u optimizando los hiperparámetros.

+++Evaluar la capacidad del modelo final con el conjunto de test para 
tener una estimación de la capacidad que tiene el modelo cuando predice 
nuevas observaciones.

+++Entrenar el modelo final con todos los datos disponibles.

'''
#111111111111111111111111111111111111111111111111111111111111111111111111111111
#Librerías
#111111111111111111111111111111111111111111111111111111111111111111111111111111

# Tratamiento de datos
# ==============================================================================
import numpy as np
import pandas as pd
from tabulate import tabulate

# Gráficos
# ==============================================================================
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib.ticker as ticker
import seaborn as sns
import statsmodels.api as sm

# Preprocesado y modelado
# ==============================================================================
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_blobs
from sklearn.metrics import euclidean_distances
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge

#Scikit-Optimize -> skopt       #pip install Scikit-Optimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt import gp_minimize
from skopt.plots import plot_convergence
import optuna

# Varios
# ==============================================================================
import os
import multiprocessing
import random
from itertools import product
from fitter import Fitter, get_common_distributions

# Configuración matplotlib
# ==============================================================================
plt.rcParams['image.cmap'] = "bwr"
#plt.rcParams['figure.dpi'] = "100"
plt.rcParams['savefig.bbox'] = "tight"
style.use('ggplot') or plt.style.use('ggplot')

# Configuración warnings
# ==============================================================================
import warnings
warnings.filterwarnings('ignore')

#222222222222222222222222222222222222222222222222222222222222222222222222222222
#Datos
#222222222222222222222222222222222222222222222222222222222222222222222222222222

'''
El set de datos SaratogaHouses del paquete mosaicData de R contiene información sobre 
el precio de 1728 viviendas situadas en Saratoga County, New York, USA en el año 2006.
Además del precio, incluye 15 variables adicionales:

price: precio de la vivienda.
lotSize: metros cuadrados de la vivienda.
age: antigüedad de la vivienda.
landValue: valor del terreno.
livingArea: metros cuadrados habitables.
pctCollege: porcentaje del vecindario con título universitario.
bedrooms: número de dormitorios.
firplaces: número de chimeneas.
bathrooms: número de cuartos de baño (el valor 0.5 hace referencia a cuartos de baño sin ducha).
rooms: número de habitaciones.
heating: tipo de calefacción.
fuel: tipo de alimentación de la calefacción (gas, electricidad o diesel).
sewer: tipo de desagüe.
waterfront: si la vivienda tiene vistas al lago.
newConstruction: si la vivienda es de nueva construcción.
centralAir: si la vivienda tiene aire acondicionado.
Pueden descargarse los datos en formato csv de SaratogaHouses.csv

El objetivo es obtener un modelo capaz de predecir el precio del alquiler.
'''
#???????????????????????carga de los datos por internet
url = (
    "https://raw.githubusercontent.com/JoaquinAmatRodrigo/Estadistica-machine-learning-python/"
    "master/data/SaratogaHouses.csv"
)
##datos = pd.read_csv(url, sep=",")

#???????????????????????carga de los datos localmente
os.chdir('D:\\Cursos\\GEM\\Diplomados\\Especializacion en estadistica y cincia de datos\\Modulo 4 Programacion con python\\ML\\')
datos = pd.read_csv(r'SaratogaHouses.csv')

# Se renombran las columnas para que sean más descriptivas
datos.columns = ["precio", "metros_totales", "antiguedad", "precio_terreno", "metros_habitables",
                 "universitarios", "dormitorios", "chimenea", "banyos", "habitaciones",
                 "calefaccion","consumo_calefacion", "desague", "vistas_lago", "nueva_construccion",
                 "aire_acondicionado"]


#333333333333333333333333333333333333333333333333333333333333333333333333333333
#Análisis exploratorio
#333333333333333333333333333333333333333333333333333333333333333333333333333333

'''
Antes de entrenar un modelo predictivo, o incluso antes de realizar cualquier cálculo con un nuevo conjunto de datos,
es muy importante realizar una exploración descriptiva de los mismos. Este proceso permite entender mejor qué información 
contiene cada variable, así como detectar posibles errores. Algunos ejemplos frecuentes son:

???Que una columna se haya almacenado con el tipo incorrecto: 
una variable numérica está siendo reconocida como texto o viceversa.

???Que una variable contenga valores que no tienen sentido: 
por ejemplo, para indicar que no se dispone del precio de una vivienda se 
introduce el valor 0 o un espacio en blanco.

???Que en una variable de tipo numérico se haya introducido una palabra 
en lugar de un número.

Además, este análisis inicial puede dar pistas sobre qué variables son adecuadas
como predictores en un modelo (más sobre esto en los siguientes apartados).
'''

print(datos.head(20))

#3.1 3.1 3.1 3.1 3.1 3.1 3.1 3.1 3.1 3.1 3.1 3.1 3.1 3.1 3.1 3.1
#3.1 Tipo de cada columna
#3.1 3.1 3.1 3.1 3.1 3.1 3.1 3.1 3.1 3.1 3.1 3.1 3.1 3.1 3.1 3.1

# ==============================================================================
# En pandas, el tipo "object" hace referencia a strings
# datos.dtypes
datos.info()

#nota:Todas las columnas tienen el tipo adecuado.

#3.2 3.2 3.2 3.2 3.2 3.2 3.2 3.2 3.2 3.2 3.2 3.2 3.2 3.2 3.2 3.2
#3.2 Número de observaciones y valores ausentes
#3.2 3.2 3.2 3.2 3.2 3.2 3.2 3.2 3.2 3.2 3.2 3.2 3.2 3.2 3.2 3.2

# Dimensiones del dataset
# ==============================================================================
print(datos.shape)

# Número de datos ausentes por variable
# ==============================================================================
print(datos.isna().sum().sort_values())

#3.3 3.3 3.3 3.3 3.3 3.3 3.3 3.3 3.3 3.3 3.3 3.3 3.3 3.3 3.3 3.3
#3.3 Variable respuesta
#3.3 3.3 3.3 3.3 3.3 3.3 3.3 3.3 3.3 3.3 3.3 3.3 3.3 3.3 3.3 3.3
'''
Cuando se crea un modelo, es muy importante estudiar la distribución 
de la variable respuesta, ya que, a fin de cuentas, es lo que interesa predecir.
La variable precio tiene una distribución asimétrica con una cola positiva debido a que,
unas pocas viviendas, tienen un precio muy superior a la media. Este tipo de distribución 
suele visualizarse mejor tras aplicar el logarítmica o la raíz cuadrada.
'''

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(6, 6))
sns.kdeplot(
    datos.precio,
    fill    = True,
    color   = "blue",
    ax      = axes[0]
)
sns.rugplot(
    datos.precio,
    color   = "blue",
    ax      = axes[0]
)
axes[0].set_title("Distribución original", fontsize = 'medium')
axes[0].set_xlabel('precio', fontsize='small') 
axes[0].tick_params(labelsize = 6)

sns.kdeplot(
    np.sqrt(datos.precio),
    fill    = True,
    color   = "blue",
    ax      = axes[1]
)
sns.rugplot(
    np.sqrt(datos.precio),
    color   = "blue",
    ax      = axes[1]
)
axes[1].set_title("Transformación raíz cuadrada", fontsize = 'medium')
axes[1].set_xlabel('sqrt(precio)', fontsize='small') 
axes[1].tick_params(labelsize = 6)

sns.kdeplot(
    np.log(datos.precio),
    fill    = True,
    color   = "blue",
    ax      = axes[2]
)
sns.rugplot(
    np.log(datos.precio),
    color   = "blue",
    ax      = axes[2]
)
axes[2].set_title("Transformación logarítmica", fontsize = 'medium')
axes[2].set_xlabel('log(precio)', fontsize='small') 
axes[2].tick_params(labelsize = 6)

fig.tight_layout()
plt.show() 

'''
Algunos modelos de machine learning y aprendizaje estadístico requieren que la variable respuesta 
se distribuya de una forma determinada. Por ejemplo, para los modelos de regresión lineal (LM), 
la distribución tiene que ser de tipo normal. Para los modelos lineales generalizados (GLM),
la distribución tiene que ser de la familia exponencial.

Existen varias librerías en python que permiten identificar a qué distribución se ajustan mejor los datos,
una de ellas es fitter. Esta librería permite ajustar cualquiera de las 80 distribuciones implementadas en scipy.
'''
distribuciones = ['cauchy', 'chi2', 'expon',  'exponpow', 'gamma',
                  'norm', 'powerlaw', 'beta', 'logistic']

fitter = Fitter(datos.precio, distributions=distribuciones)
fitter.fit()
print(fitter.summary(Nbest=10, plot=False))

#3.4 3.4 3.4 3.4 3.4 3.4 3.4 3.4 3.4 3.4 3.4 3.4 3.4 3.4 3.4 3.4
# Variables numéricas
#3.4 3.4 3.4 3.4 3.4 3.4 3.4 3.4 3.4 3.4 3.4 3.4 3.4 3.4 3.4 3.4

# ==============================================================================
print(datos.select_dtypes(include=['float64', 'int']).describe())


# Gráfico de distribución para cada variable numérica
# ==============================================================================
# Ajustar número de subplots en función del número de columnas
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(9, 5))
axes = axes.flat
columnas_numeric = datos.select_dtypes(include=['float64', 'int']).columns
columnas_numeric = columnas_numeric.drop('precio')

for i, colum in enumerate(columnas_numeric):
    sns.histplot(
        data     = datos,
        x        = colum,
        stat     = "count",
        kde      = True,
        color    = (list(plt.rcParams['axes.prop_cycle'])*2)[i]["color"],
        line_kws = {'linewidth': 2},
        alpha    = 0.3,
        ax       = axes[i]
    )
    axes[i].set_title(colum, fontsize = 7, fontweight = "bold")
    axes[i].tick_params(labelsize = 6)
    axes[i].set_xlabel("")
    
    
fig.tight_layout()
plt.subplots_adjust(top = 0.9)
fig.suptitle('Distribución variables numéricas', fontsize = 10, fontweight = "bold");

plt.show()

'''
La variable chimenea, aunque es de tipo numérico, apenas toma unos pocos valores y la gran
mayoría de observaciones pertenecen a solo dos de ellos. En casos como este, suele ser conveniente
tratar la variable como cualitativa.
'''

# Valores observados de chimenea
# ==============================================================================
print(datos.chimenea.value_counts())


# Se convierte la variable chimenea tipo string
# ==============================================================================
datos.chimenea = datos.chimenea.astype("str")

'''
Como el objetivo del estudio es predecir el precio de las viviendas, el análisis de cada
variable se hace también en relación a la variable respuesta precio. Analizando los datos de esta
forma, se pueden empezar a extraer ideas sobre qué variables están más relacionadas con el precio 
y de qué forma.
'''

# Gráfico de distribución para cada variable numérica
# ==============================================================================
# Ajustar número de subplots en función del número de columnas
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(9, 5))
axes = axes.flat
columnas_numeric = datos.select_dtypes(include=['float64', 'int']).columns
columnas_numeric = columnas_numeric.drop('precio')

for i, colum in enumerate(columnas_numeric):
    sns.regplot(
        x           = datos[colum],
        y           = datos['precio'],
        color       = "gray",
        marker      = '.',
        scatter_kws = {"alpha":0.4},
        line_kws    = {"color":"r","alpha":0.7},
        ax          = axes[i]
    )
    axes[i].set_title(f"precio vs {colum}", fontsize = 7, fontweight = "bold")
    #axes[i].ticklabel_format(style='sci', scilimits=(-4,4), axis='both')
    axes[i].yaxis.set_major_formatter(ticker.EngFormatter())
    axes[i].xaxis.set_major_formatter(ticker.EngFormatter())
    axes[i].tick_params(labelsize = 6)
    axes[i].set_xlabel("")
    axes[i].set_ylabel("")

# Se eliminan los axes vacíos
for i in [8]:
    fig.delaxes(axes[i])
    
fig.tight_layout()
plt.subplots_adjust(top=0.9)
fig.suptitle('Correlación con precio', fontsize = 10, fontweight = "bold");
plt.show()

#3.5 3.5 3.5 3.5 3.5 3.5 3.5 3.5 3.5 3.5 3.5 3.5 3.5 3.5 3.5 3.5
#Correlación variables numéricas
#3.5 3.5 3.5 3.5 3.5 3.5 3.5 3.5 3.5 3.5 3.5 3.5 3.5 3.5 3.5 3.5
'''
Algunos modelos (LM, GLM, ...) se ven perjudicados si incorporan predictores altamente 
correlacionados. Por esta razón, es conveniente estudiar el grado de correlación 
entre las variables disponibles.
'''

# Correlación entre columnas numéricas
# ==============================================================================
def tidy_corr_matrix(corr_mat):
    '''
    Función para convertir una matrix de correlación de pandas en formato tidy
    '''
    corr_mat = corr_mat.stack().reset_index()
    corr_mat.columns = ['variable_1','variable_2','r']
    corr_mat = corr_mat.loc[corr_mat['variable_1'] != corr_mat['variable_2'], :]
    corr_mat['abs_r'] = np.abs(corr_mat['r'])
    corr_mat = corr_mat.sort_values('abs_r', ascending=False)
    
    return(corr_mat)

corr_matrix = datos.select_dtypes(include=['float64', 'int']).corr(method='pearson')
print(tidy_corr_matrix(corr_matrix).head(10))

# Heatmap matriz de correlaciones
# ==============================================================================
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))

sns.heatmap(
    corr_matrix,
    annot     = True,
    cbar      = False,
    annot_kws = {"size": 6},
    vmin      = -1,
    vmax      = 1,
    center    = 0,
    cmap      = sns.diverging_palette(20, 220, n=200),
    square    = True,
    ax        = ax
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation = 45,
    horizontalalignment = 'right',
)

ax.tick_params(labelsize = 8)
plt.show()

#3.6 3.6 3.6 3.6 3.6 3.6 3.6 3.6 3.6 3.6 3.6 3.6 3.6 3.6 3.6 3.6
#Variables cualitativas
#3.6 3.6 3.6 3.6 3.6 3.6 3.6 3.6 3.6 3.6 3.6 3.6 3.6 3.6 3.6 3.6
# Variables cualitativas (tipo object)
# ==============================================================================
print(datos.select_dtypes(include=['object']).describe())

# Gráfico para cada variable cualitativa
# ==============================================================================
# Ajustar número de subplots en función del número de columnas
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(9, 5))
axes = axes.flat
columnas_object = datos.select_dtypes(include=['object']).columns

for i, colum in enumerate(columnas_object):
    datos[colum].value_counts().plot.barh(ax = axes[i])
    axes[i].set_title(colum, fontsize = 7, fontweight = "bold")
    axes[i].tick_params(labelsize = 6)
    axes[i].set_xlabel("")

# Se eliminan los axes vacíos
for i in [7, 8]:
    fig.delaxes(axes[i])
    
fig.tight_layout()
plt.subplots_adjust(top=0.9)
fig.suptitle('Distribución variables cualitativas',
             fontsize = 10, fontweight = "bold");
plt.show()

'''
Si alguno de los niveles de una variable cualitativa tiene muy pocas observaciones en 
comparación a los otros niveles, puede ocurrir que, durante la validación cruzada o 
bootstrapping, algunas particiones no contengan ninguna observación de dicha clase 
(varianza cero), lo que puede dar lugar a errores. En estos casos, suele ser conveniente:

---Eliminar las observaciones del grupo minoritario si es una variable multiclase.
---Eliminar la variable si solo tiene dos niveles.
---Agrupar los niveles minoritarios en un único grupo.
---Asegurar que, en la creación de las particiones, todos los grupos estén representados 
en cada una de ellas.

Para este caso, hay que tener precaución con la variable chimenea.
Se unifican los niveles de 2, 3 y 4 en un nuevo nivel llamado "2_mas".
'''

print(datos.chimenea.value_counts().sort_index())

dic_replace = {'2': "2_mas",
               '3': "2_mas",
               '4': "2_mas"}

datos['chimenea'] = (
    datos['chimenea']
    .map(dic_replace) 
    .fillna(datos['chimenea'])
)

print(datos.chimenea.value_counts().sort_index())

# Gráfico relación entre el precio y cada cada variables cualitativas
# ==============================================================================
# Ajustar número de subplots en función del número de columnas
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(9, 5))
axes = axes.flat
columnas_object = datos.select_dtypes(include=['object']).columns

for i, colum in enumerate(columnas_object):
    sns.violinplot(
        x     = colum,
        y     = 'precio',
        data  = datos,
        color = "white",
        ax    = axes[i]
    )
    axes[i].set_title(f"precio vs {colum}", fontsize = 7, fontweight = "bold")
    axes[i].yaxis.set_major_formatter(ticker.EngFormatter())
    axes[i].tick_params(labelsize = 6)
    axes[i].set_xlabel("")
    axes[i].set_ylabel("")

# Se eliminan los axes vacíos
for i in [7, 8]:
    fig.delaxes(axes[i])
    
fig.tight_layout()
plt.subplots_adjust(top=0.9)
fig.suptitle('Distribución del precio por grupo', fontsize = 10, fontweight = "bold");
plt.show()

#3.7 3.7 3.7 3.7 3.7 3.7 3.7 3.7 3.7 3.7 3.7 3.7 3.7 3.7 3.7 3.7
#División train y test
#3.7 3.7 3.7 3.7 3.7 3.7 3.7 3.7 3.7 3.7 3.7 3.7 3.7 3.7 3.7 3.7
'''
Evaluar la capacidad predictiva de un modelo consiste en comprobar cómo de próximas 
son sus predicciones a los verdaderos valores de la variable respuesta. Para poder 
cuantificarlo de forma correcta, se necesita disponer de un conjunto de observaciones,
de las que se conozca la variable respuesta, pero que el modelo no haya "visto", es decir,
que no hayan participado en su ajuste. Con esta finalidad, se dividen los datos disponibles 
en un conjunto de entrenamiento y un conjunto de test. 
El tamaño adecuado de las particiones depende en gran medida de la cantidad de datos 
disponibles y la seguridad que se necesite en la estimación del error, 80%-20% suele 
dar buenos resultados. El reparto debe hacerse de forma aleatoria o aleatoria-estratificada.
'''
# Reparto de datos en train y test
# ==============================================================================
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
                                        datos.drop('precio', axis = 'columns'),
                                        datos['precio'],
                                        train_size   = 0.8,
                                        random_state = 1234,
                                        shuffle      = True
                                    )

print("Partición de entrenamento")
print("-----------------------")
print(y_train.describe())

print("Partición de test")
print("-----------------------")
print(y_test.describe())

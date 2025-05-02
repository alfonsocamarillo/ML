'''
https://cienciadedatos.net/documentos/py35-redes-neuronales-python
Ejemplo clasificación
En este primer ejemplo se muestra cómo, dependiendo de la arquitectura (capas ocultas y tamaño de las mismas), un modelo basado en redes neuronales puede aprender funciones no lineales con gran facilidad. Para evitar problemas de overfitting, es importante identificar la combinación de hiperparámetros que consigue un equilibrio adecuado en el aprendizaje. Se trata de un ejemplo muy sencillo, cuyo objetivo es que el lector se familiarice con flexibilidad que tienen este tipo de modelos y en cómo realizar una búsqueda de hiperparámetros mediante grid search y validación cruzada.
'''
# ==============================================================================
#Librerías
#Las librerías utilizadas en este ejemplo son:
# ==============================================================================

# Tratamiento de datos
# ==============================================================================
import numpy as np
import pandas as pd

# Gráficos
# ==============================================================================
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline
plt.style.use('fivethirtyeight')

# Modelado
# ==============================================================================
from sklearn.datasets import make_blobs
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
import multiprocessing

# Configuración warnings
# ==============================================================================
import warnings
warnings.filterwarnings('ignore')

#Datos
#Se simulan observaciones en dos dimensiones, pertenecientes a tres grupos, cuya separación no es perfecta.
# ==============================================================================
# Datos simulados
# ==============================================================================
X, y = make_blobs(
        n_samples    = 500, 
        n_features   = 2, 
        centers      = 3, 
        cluster_std  = 1.2, 
        shuffle      = True, 
        random_state = 0
    )

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
for i in np.unique(y):
    ax.scatter(
        x = X[y == i, 0],
        y = X[y == i, 1], 
        c = plt.rcParams['axes.prop_cycle'].by_key()['color'][i],
        marker    = 'o',
        edgecolor = 'black', 
        label= f"Grupo {i}"
    )
    
ax.set_title('Datos simulados')
ax.legend()
plt.show()


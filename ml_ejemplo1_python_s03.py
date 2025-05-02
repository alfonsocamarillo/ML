'''
Modelos supervisados

Regression:
Este modelo supervisado se usa habitualmente cuando se pretende predecir un valor numérico. 

Responde a preguntas del tipo: 
¿cuántos productos venderemos el mes que viene?, 
¿qué beneficios tendremos el próximo año?, etc.

Este tipo de algoritmos intenta cuantificar un valor basándose 
en el comportamiento previo de los datos.
'''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

os.chdir('D:\\Cursos\\GEM\\Diplomados\\Especializacion en estadistica y cincia de datos\\Modulo 4 Programacion con python\\ML')
df = pd.read_csv(r'costs-purchases.csv')

x = np.reshape(np.array(df['Spend']), (-1, 1))
y = np.reshape(np.array(df['Purchases']), (-1, 1))
lm = LinearRegression()
lm.fit(x, y)
y_predicted = lm.predict(x)

plt.scatter(x, y, s=10)
plt.xlabel('Spend')
plt.ylabel('Purchases')
plt.plot(x, y_predicted, color='r')
plt.show()

rmse = mean_squared_error(y, y_predicted)
r2 = r2_score(y, y_predicted)
print('Slope:', lm.coef_)
print('Intercept:', lm.intercept_)
print('Root mean squared error: ', rmse)
print('R2 score: ', r2)

'''
Los resultados proporcionados parecen provenir de un modelo de regresión lineal simple. 
Aquí hay una interpretación de cada uno de los valores:

+++Pendiente (Slope): Slope. La fórmula en la que se basa el algoritmo Linear Regression es y = mx + b. 
En este caso, el Slope (inclinación de la línea) hace referencia a la letra m.

    Es el coeficiente de la variable independiente en el modelo de regresión. 
    En este caso, parece ser aproximadamente 0.1347. 
    Esto significa que, en promedio, por cada unidad de aumento en la variable independiente, 
    se espera que la variable dependiente aumente en 0.1347 unidades.

+++Intercepto (Intercept): Intercept. Es el valor constante b. Tanto el slope como el intercept representan
en este caso la relación lineal entre 2 variables.

    Es el valor de la variable dependiente cuando la variable independiente es cero. 
    En este caso, parece ser aproximadamente -1337.0093. 
    Indica el valor esperado de la variable dependiente cuando todas las variables 
    independientes son cero.

+++Error cuadrático medio (Root mean squared error): Root Mean Squared error (RMSE). Representa la desviación estándar 
de los residuos o lo que es lo mismo cuánto varía cada dato con respecto a la línea de regresión. 
Como el resultado dependerá del valor de los datos, se puede utilizar la fórmula (rmse / avg(y)) * 100%) 
para aceptar o no el modelo generado. Se suele considerar < 10% como óptimo.

    Es una medida de la diferencia entre los valores predichos por el modelo y los valores observados. 
    En este caso, el error cuadrático medio parece ser aproximadamente 2407734.622. 
    Cuanto menor sea este valor, mejor será el ajuste del modelo a los datos observados.

+++Coeficiente de determinación (R2 score): R2 error. Mide la relación existente entre el modelo generado 
y la variable dependiente. No hay un valor concreto a partir del cual se puede considerar como óptimo, 
sin embargo a partir de 60% (0,6) ya se empieza a entender como una correlación alta. 
Aunque esto dependerá de otros muchos factores y ámbitos de estudio.

    Es una medida de qué tan bien el modelo se ajusta a los datos observados. 
    Varía entre 0 y 1, donde 1 indica un ajuste perfecto. 
    En este caso, el coeficiente R cuadrado parece ser aproximadamente 0.6249. 
    Esto sugiere que alrededor del 62.49% de la variabilidad en la variable dependiente 
    puede explicarse por la variable independiente en el modelo.

'''
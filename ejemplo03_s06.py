'''
Ejemplo en Python con gráficos  de Funciones de activación en Estadística
Claro, aquí tienes un ejemplo en Python que muestra gráficamente algunas de las funciones de activación comunes en estadística utilizando la biblioteca Matplotlib:
'''
import numpy as np
import matplotlib.pyplot as plt

# Definir las funciones de activación
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Restar el máximo para evitar problemas de overflow
    return exp_x / exp_x.sum(axis=0)

# Definir el rango de valores para x
x = np.linspace(-5, 5, 100)

# Calcular los valores de las funciones de activación
y_sigmoid = sigmoid(x)
y_relu = relu(x)
y_tanh = tanh(x)
y_softmax = softmax(x)

# Crear subplots
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# Graficar la función de activación sigmoide
axs[0, 0].plot(x, y_sigmoid, label='Sigmoide', color='b')
axs[0, 0].set_title('Función Sigmoide')
axs[0, 0].legend()

# Graficar la función de activación ReLU
axs[0, 1].plot(x, y_relu, label='ReLU', color='r')
axs[0, 1].set_title('Función ReLU')
axs[0, 1].legend()

# Graficar la función de activación tanh
axs[1, 0].plot(x, y_tanh, label='Tanh', color='g')
axs[1, 0].set_title('Función Tanh')
axs[1, 0].legend()

# Graficar la función de activación softmax
axs[1, 1].plot(x, y_softmax, label='Softmax', color='purple')
axs[1, 1].set_title('Función Softmax')
axs[1, 1].legend()

# Ajustar el diseño
plt.tight_layout()

# Mostrar las gráficas
plt.show()

'''
Este código generará un gráfico con cuatro subgráficos, cada uno mostrando una función de activación diferente: sigmoide, ReLU, tanh y softmax. Puedes experimentar con diferentes rangos de valores para x y ver cómo cambian las funciones de activación.
'''
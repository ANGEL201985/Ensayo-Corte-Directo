import pandas as pd
import numpy as np
import matplotlib.pyplot as plt #  Importa la submódulo pyplot de la librería Matplotlib, que se utiliza para crear gráficos.
from sklearn.linear_model import LinearRegression # Importa la clase LinearRegression del módulo linear_model de la librería scikit-learn. Esta clase se utiliza para realizar regresión lineal.
from IPython.display import display  # Importar display

#Datos iniciales Generales
area_inicial = 36
longitud = 60 

#Ecuacion de anillo de corte
coeficiente_termino_lineal = 0.357
termino_independiente = 0.464

#Lectura de la deformacion tangencial
deformacion_tangencial= np.array([0.00, 0.03, 0.06, 0.12, 0.18, 0.30, 0.45, 0.60, 0.75, 0.90, 1.05, 1.20, 1.50, 1.80, 2.10, 2.40, 2.70, 3.00, 3.60, 4.20, 4.80, 5.40, 6.00])

#Calculando el area corregido
area_corregido = (longitud - deformacion_tangencial)*longitud/100
#print(area_corregido)

# Datos de la muestra 1
carga_1 = 1.8
esfuerzo_normal_1 = carga_1*10/area_inicial
#print(esfuerzo_normal_1)

#Realizaremos los calculos para la muestra 1
dial_carga_1 = np.array([0, 13, 16, 18, 20, 23, 26, 28, 29, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 43])
fuerza_corte_1 = np.around(dial_carga_1*coeficiente_termino_lineal + termino_independiente, 2)

# Asegurando que el primer valor sea cero si el primer valor de dial_carga_1 es cero
if dial_carga_1[0] == 0:
    fuerza_corte_1[0] = 0
#print(fuerza_corte)

#Calculamos al Esfuerzo Normal para las areas corregidas
esfuerzo_normal_1_corregido = np.around(carga_1/area_corregido,3)

#Calculamos el esfuerzo de corte
esfuerzo_corte_1 = np.around(fuerza_corte_1/area_corregido,3)
esfuerzo_corte_1_max = np.max(esfuerzo_corte_1)
#print(esfuerzo_corte)

#Almacenando en una tabla los valores y calculos

datos_1 = {'Deformacion Tangencial (mm)':deformacion_tangencial, 'Dial de carga': dial_carga_1, 'Fuerza de Corte (kg)':fuerza_corte_1, 'Area Corregida (cm2)': area_corregido, 'Esfuerzo Normal (kg/cm2)': esfuerzo_normal_1_corregido, 'Esfuerzo de Corte (kg/cm2)':esfuerzo_corte_1}

muestra_1 = pd.DataFrame(datos_1)
print(muestra_1)

# Datos de la muestra 2
carga_2 = 3.6
esfuerzo_normal_2 = carga_2*10/area_inicial
#print(esfuerzo_normal_2)

#Realizaremos los calculos para la muestra 2
dial_carga_2 = np.array([0, 24, 28, 32, 35, 39, 43, 48, 53, 58, 62, 63, 64, 65, 66, 67, 69, 71, 72, 73, 74, 74, 74])
fuerza_corte_2 = np.around(dial_carga_2*coeficiente_termino_lineal + termino_independiente, 2)

# Asegurando que el primer valor sea cero si el primer valor de dial_carga_1 es cero
if dial_carga_2[0] == 0:
    fuerza_corte_2[0] = 0
#print(fuerza_corte)

#Calculamos al Esfuerzo Normal para las areas corregidas
esfuerzo_normal_2_corregido = np.around(carga_2/area_corregido,3)

#Calculamos el esfuerzo de corte
esfuerzo_corte_2 = np.around(fuerza_corte_2/area_corregido,3)
esfuerzo_corte_2_max = np.max(esfuerzo_corte_2)
#print(esfuerzo_corte)

#Almacenando en una tabla los valores y calculos
datos_2 = {'Deformacion Tangencial (mm)':deformacion_tangencial, 'Dial de carga': dial_carga_2, 'Fuerza de Corte (kg)':fuerza_corte_2, 'Area Corregida (cm2)': area_corregido, 'Esfuerzo Normal (kg/cm2)': esfuerzo_normal_2_corregido, 'Esfuerzo de Corte (kg/cm2)':esfuerzo_corte_2}

muestra_2 = pd.DataFrame(datos_2)
print(muestra_2)


# Datos de la muestra 3
carga_3 = 7.2
esfuerzo_normal_3 = carga_3*10/area_inicial
#print(esfuerzo_normal_3)

#Realizaremos los calculos para la muestra 3
dial_carga_3 = np.array([0, 28, 38, 45, 55, 65, 75, 85, 90, 93, 96, 98, 101, 105, 108, 109, 110, 111, 112, 113, 114, 114, 114])
fuerza_corte_3 = np.around(dial_carga_3*coeficiente_termino_lineal + termino_independiente, 2)

# Asegurando que el primer valor sea cero si el primer valor de dial_carga_1 es cero
if dial_carga_3[0] == 0:
    fuerza_corte_3[0] = 0
#print(fuerza_corte)

#Calculamos al Esfuerzo Normal para las areas corregidas
esfuerzo_normal_3_corregido = np.around(carga_3/area_corregido,3)

#Calculamos el esfuerzo de corte
esfuerzo_corte_3 = np.around(fuerza_corte_3/area_corregido,3)
esfuerzo_corte_3_max = np.max(esfuerzo_corte_3)
#print(esfuerzo_corte)

#Almacenando en una tabla los valores y calculos
datos_3 = {'Deformacion Tangencial (mm)':deformacion_tangencial, 'Dial de carga': dial_carga_3, 'Fuerza de Corte (kg)':fuerza_corte_3, 'Area Corregida (cm2)': area_corregido, 'Esfuerzo Normal (kg/cm2)': esfuerzo_normal_2_corregido, 'Esfuerzo de Corte (kg/cm2)':esfuerzo_corte_3}

muestra_3 = pd.DataFrame(datos_3)
print(muestra_3)


# Ajustar opciones de visualización con metodo set_option de Pandas
pd.set_option('display.max_columns', None) #display.max_columns': Es la opción que controla el número máximo de columnas que se mostrarán al imprimir un DataFrame. None: Al establecer este valor, se indica que no hay límite en el número de columnas a mostrar. Es decir, pandas mostrará todas las columnas del DataFrame sin importar cuántas sean. En resumen, esta línea de código asegura que, al imprimir un DataFrame, se mostrarán todas las columnas disponibles en lugar de truncarlas a un número predeterminado.
pd.set_option('display.max_colwidth', None)# display.max_colwidth': Es la opción que controla el ancho máximo de las columnas cuando se imprimen valores en un DataFrame. None: Al establecer este valor, se indica que no hay límite en el ancho de las columnas. En resumen, esta línea de código asegura que, al imprimir un DataFrame, se mostrará el contenido completo de cada celda sin importar cuán largo sea el texto
pd.set_option('display.width', 1000)#'display.width': Es la opción que controla el ancho total de la salida de texto cuando se imprime un DataFrame en la consola. 1000: Al establecer este valor, se indica que el ancho máximo para la representación tabular del DataFrame será de 1000 caracteres.

# Mostrar las tablas estilizadas
muestra_1_estilo = muestra_1.style.set_table_attributes('style="width:100%"')#Este código crea una versión estilizada de muestra_1 llamada muestra_1_styled. Utiliza el método style.set_table_attributes() para establecer los atributos de estilo de la tabla, en este caso, el ancho de la tabla se establece en 100%.
display(muestra_1_estilo)#Luego, el DataFrame estilizado muestra_1_styled se muestra en el entorno de Python, es decir se mostrara la tabla como si fuera un print.

muestra_2_stilo = muestra_2.style.set_table_attributes('style="width:100%"')
display(muestra_2_stilo)

muestra_3_stilo = muestra_3.style.set_table_attributes('style="width:100%"')
display(muestra_3_stilo)


"""
#GRAFICANDO LA DEFORMACION TANGENCIAL(mm) vs ESFUERZO DE CORTE (kg/cm2)

x1 = deformacion_tangencial
y1 = esfuerzo_corte_1
plt.scatter(x1,y1)
plt.plot(x1,y1, label='Muestra 1') # El plot nos permite trazar rectas entre los puntos

x2 = deformacion_tangencial
y2 = esfuerzo_corte_2
plt.scatter(x2,y2)
plt.plot(x2,y2, label='Muestra 2')# El plot nos permite trazar rectas entre los puntos

x3 = deformacion_tangencial
y3 = esfuerzo_corte_3
plt.scatter(x3,y3)
plt.plot(x3,y3, label='Muestra 3')# El plot nos permite trazar rectas entre los puntos

plt.xlabel('Deformacion Tangencial (mm)')
plt.ylabel('Esfuerzo de Corte (kg/cm2)')
plt.grid()
plt.title('Deformacion Tangencial vs Esfuerzo de Corte')
plt.legend()# Nos permite mostrar los label de cada plot

# Ajustar los límites de los ejes, esto nos permite que la coordenada (0,0) comience en la parte inferior del grafico
plt.xlim(0, max(deformacion_tangencial) + 0.5)#0: Es el límite inferior del eje x, lo que significa que la gráfica del eje x empezará en 0. max(deformacion_tangencial) + 0.5: Calcula el límite superior del eje x. Toma el valor máximo de la lista o array deformacion_tangencial y le añade 0.5 unidades. Esto se hace para asegurar que haya un pequeño margen adicional en el gráfico, evitando que el punto máximo se quede justo en el borde de la gráfica.
plt.ylim(0, max(esfuerzo_corte_1.max(), esfuerzo_corte_2.max(), esfuerzo_corte_3.max()) + 0.2)# 0: Es el límite inferior del eje y, lo que significa que la gráfica del eje y empezará en 0. max(esfuerzo_corte_1.max(), esfuerzo_corte_2.max(), esfuerzo_corte_3.max()) + 0.5: Calcula el límite superior del eje y. Toma el valor máximo entre los valores máximos de tres arrays o listas (esfuerzo_corte_1, esfuerzo_corte_2, esfuerzo_corte_3) y le añade 0.5 unidades. Esto también se hace para asegurar que haya un pequeño margen adicional en el gráfico, evitando que el punto máximo se quede justo en el borde de la gráfica.

plt.show()

"""
#GRAFICANDO EL ESFUERZO TANGENCIAL (kg/cm2) vs ESFUERZO DE CORTE (kg/cm2)
x4 = np.array([esfuerzo_normal_1, esfuerzo_normal_2, esfuerzo_normal_3]).reshape(-1, 1)
y4 = np.array([esfuerzo_corte_1_max, esfuerzo_corte_2_max, esfuerzo_corte_3_max])

# Crear el modelo de regresión lineal
modelo = LinearRegression()# Crea una instancia de la clase LinearRegression y la guarda en modelo.
modelo.fit(x4, y4)# Ajusta el modelo de regresión lineal a los datos.

# Coeficientes de la regresión lineal
pendiente = modelo.coef_[0]# coef_ es un atributo que pertenece a los modelos de regresión de scikit-learn es un array que contiene los coeficientes de las características de entrada (pendientes) del modelo ajustado. En este caso, como sólo hay una característica (esfuerzo normal), model.coef_ es un array de un solo elemento. model.coef_[0] obtiene la pendiente de la línea de regresión.
print('La pendiente es:', pendiente)
interseccion = modelo.intercept_ # intercept_ es un atributo en los modelos de regresión de scikit-learn en Python. Este atributo contiene el término independiente (también conocido como intercepto) de la ecuación de regresión, que representa el valor de la variable dependiente cuando todas las variables independientes son cero.

# Calcular el angulo de friccion y la cohesion.
cohesion = interseccion
angulo_radianes = np.arctan(pendiente)
angulo_friccion = np.degrees(angulo_radianes)


# Generar puntos para la línea de regresión
x_rango = np.linspace(min(x4), max(x4), 100).reshape(-1, 1)#np.linspace es una función de NumPy que genera 100 valores igualmente espaciados entre el valor mínimo (min(x4)) y el valor máximo (max(x4)) de x4. Esto crea un array de 100 puntos que cubren el rango de los datos de entrada x4. reshape(-1, 1) convierte el array de 1D de 100 elementos en una matriz 2D con 100 filas y 1 columna.
print('los valores de esfuerzo normal son:', x_rango)
y_prediccion = modelo.predict(x_rango)# predict es un método del objeto de regresión lineal de scikit learn que predice los valores de salida (esfuerzo de corte, en este caso) para las entradas proporcionadas (x_range). Utiliza los coeficientes del modelo de regresión (pendiente e intersección) que se ajustaron anteriormente con model.fit.
print('los valores de prediccion son:', y_prediccion)


# Graficar los puntos y la línea de regresión
plt.scatter(x4, y4, color='green')
plt.plot(x_rango, y_prediccion, color='red', label=f'Ecuacion: y = {pendiente:.2f}x + {interseccion:.2f}')

plt.xlabel('Esfuerzo Normal (kg/cm2)')
plt.ylabel('Esfuerzo de Corte (kg/cm2)')
plt.grid()

# Agregar leyenda con el valor de la máxima densidad seca y el contenido óptimo de humedad, con bbox agregamos estilos a la leyenda. El "transform=plt.gca().transAxes" especifica que las coordenadas (0.05, 0.90), están en el sistema de coordenadas de los ejes, no en el sistema de coordenadas de los datos del gráfico. plt.gca(): Esta función devuelve el objeto Axes actual, el cual es la región del gráfico donde se plotean los datos. pad es padding o relleno y round hace que los bordes de la caja que contiene el texto sean redondos.
plt.text(0.03, 0.90,f'Angulo de friccion(φ): {angulo_friccion:.2f}°\nCohesion: {cohesion:.2f} kg/cm2', horizontalalignment='left', verticalalignment='top', transform=plt.gca().transAxes,bbox=dict(facecolor='green', alpha=0.5, edgecolor='black', boxstyle='round,pad=0.5'))

plt.title('Esfuerzo Normal vs Esfuerzo de Corte')
plt.legend()
plt.show()





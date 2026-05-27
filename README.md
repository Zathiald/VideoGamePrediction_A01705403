# VideoGamePrediction_A01705403
VIdeoGamePrediction Model por Samir Baidon Pardo

Este proyecto es parte del curso "Desarrollo de aplicaciones avanzadas de ciencias computacionales", el propósito de este modelo es ser entrenado para que al ingresarle los siguientes parámetros:
- Platform
- Genre
- Publisher
- Rating

Entonces sea capaz de predecir:
- Ventas en Norteamérica
- Ventas en Europa
- Ventas en Japón
- Ventas Globales
- Score de Criticos

Esto con el fin de tener un modelo con el cual se pueda predecir el rendimiento de futuros juegos.

Para este modelo se tienen dos opciones de datasets principales que contienen los datos necesarios:

## 📂 Datasets Utilizados

| Dataset | Descripción | Registros | Link |
|---|---|---|---|
| 🎮 **VideoGames Sales** | Dataset creado por **Eugenio M.** que contiene información de ventas de videojuegos, incluyendo los parámetros y salidas utilizados en esta documentación. | 7,113 datos | [Ver Dataset](https://www.kaggle.com/datasets/migeruj/videogames-predictive-model) |
| 📊 **Video Game Sales** | Dataset creado por **Siddharth Vora** con información detallada de ventas de videojuegos y además imágenes del empaque físico de cada juego. | 39,798 datos | [Ver Dataset](https://www.kaggle.com/datasets/siddharth0935/video-game-sales?select=vgchartz-2024.csv) |

Como se planea utilizar estos datasets para entrenar un modelo de Machine Learning, primero fue necesario realizar un proceso de preparación y limpieza de los datos. 

Para ello, se eliminaron registros con valores nulos, se transformaron las columnas numéricas de ventas al formato adecuado y se convirtieron algunas variables categóricas a valores numéricos para que pudieran ser interpretadas por el modelo. 

Posteriormente, el dataset se dividió en conjuntos de entrenamiento y prueba, la división es la siguiente:

- 80% (5689 datos) Training 
- 20% (1423 datos) Testing

Finalmente se aplicó un escalamiento a las variables objetivo con el fin de normalizar los datos y mejorar el rendimiento del entrenamiento, este escalamiento consiste en transformar los datos para que todas las variables tengan una escala similar, evitando que algunas columnas con valores más grandes influyan más que otras durante el entrenamiento del modelo, para esto las columnas que fueron escaladas son:

| Variable | Tipo de Variable | Descripción |
|---|---|---|
| `NA_Sales` | Numérica Continua | Representa las ventas del videojuego en Norteamérica. Los valores están expresados en millones de copias vendidas. |
| `EU_Sales` | Numérica Continua | Representa las ventas del videojuego en Europa. Los valores están expresados en millones de copias vendidas. |
| `JP_Sales` | Numérica Continua | Representa las ventas del videojuego en Japón. Los valores están expresados en millones de copias vendidas. |
| `Other_Sales` | Numérica Continua | Representa las ventas del videojuego en otras regiones distintas de Norteamérica, Europa y Japón. |
| `Global_Sales` | Numérica Continua | Representa las ventas globales totales del videojuego, considerando todas las regiones. |
| `Critic_Score_Class` | Categórica Ordinal | Clasificación de la crítica del videojuego agrupada en categorías. Esta variable fue transformada a valores numéricos utilizando `LabelEncoder` para poder ser utilizada por el modelo de Machine Learning. |

El resto de las columnas no fueron escaladas ya que estás nos serviran de parámetros de entrada a la hora de hacer la predicción de datos.

<img width="980" height="452" alt="image" src="https://github.com/user-attachments/assets/99e437a3-fa8f-464e-9b45-cceffc245acd" />

### Antes del Escalamiento

La gráfica de la izquierda muestra los datos originales de ventas y clasificación crítica. Se puede observar que algunas variables, como `Global_Sales`, tienen valores mucho más grandes que otras, generando diferencias importantes en las escalas.

Además, aparecen muchos puntos fuera de las cajas, los cuales representan videojuegos con ventas extremadamente altas en comparación con el resto.

### Después del Escalamiento

La gráfica de la derecha muestra las mismas variables después de aplicar `StandardScaler`. En este caso, todas las variables fueron normalizadas para tener una escala similar alrededor de 0.

El escalamiento reduce las diferencias de magnitud entre columnas, permitiendo que el modelo de Machine Learning procese los datos de forma más equilibrada.

Gracias a esto:
- Ninguna variable domina a las demás por tener valores más grandes.
- El entrenamiento del modelo es más estable.
- Algunos algoritmos pueden converger más rápido y obtener mejores resultados.

## Selección del Modelo
Para poder seleccionar nuestro modelo una vez que se ha preprocesado los datos, es importante mantenerse informado de otra soluciones y posibles opciones para la creación de nuestra solución, en la búsqueda de problemas similares, se encotnraron estos dos papeles escritos por la Universidad Galgotias y la Universidad Capital de China, respectivamente.

| Título | Descripción | Link |
|---|---|---|
| 📃 **Machine Learning Models-Based Video Game Sales Prediction** | Paper escrito por **Kaiyue Wu** en la Universidad Capital de China, en donde a través de datos de ventas globales se predicen los valores de ventas de videojuegos. | [Ver Paper](https://www.scitepress.org/Papers/2024/132058/132058.pdf) |
| 📜 **Video game sales prediction model using regression model** | Papel escrito por **Affana** , **Sourabh Vishwakarmab** y el **Dr. Santosh Kumarc** en la  Universidad Galgotias, en donde a través de User y Critic count, con un modelo de regresión lineal se trata de predecir ventas de videojuegos. | [Ver Paper]([https://www.kaggle.com/datasets/siddharth0935/video-game-sales?select=vgchartz-2024.csv](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4935036) |

Un factor que se tiene muy en común en estos papeles es el uso del modelo RandomForest, este modelo es un algoritmo de Machine Learning supervisado que combina múltiples árboles de decisión para crear un modelo predictivo, este algoritmo se puede utilizar tanto para regresión como clasificación, perfecto para nuestro problema en donde debemos realizar predicciones numéricas y clasificativas.

## Implementación del Modelo
Ahora con nuestro modelo seleccionado es momento de implementarlo, para esto utilizaremos la librería de sklearn.ensemble, una vez importada la librería se crea el modelo de la siguiente manera:

```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    oob_score= True
)

model.fit(X_train, y_train_scaled)
```
Para esta primera versión del modelo seleccionaremos los siguientes parámetros con el fin de tener una versión base que va empezando a predecir y clasificar nuestros valores:
- n_estimators=100: Número de árboles de decisión en el bosque.
- random_state=42: Asegura resultados reproducibles.
- oob_score=True: Utilizamos la métrica out-of-the-bag para estimar el rendimiento del modelo.

Ahora con nuestro modelo implementado y nuestro dataset limpio, es momento de entrenarlo y obtener resultados.

## Métricas Utilizadas en el Modelo

Pero para evaluar el rendimiento del modelo de Machine Learning, se deben conocer qué métricas, tanto de regresión como de clasificación, debemos utilizar, esto debido a que el modelo predice variables numéricas de ventas y una variable categórica correspondiente a la clasificación de críticas.

---

### Métricas de Regresión
Estas métricas fueron escogidas en base a los papeles en donde indican explicitamente cómo utilizan MAE para analizar sus resultados de regresión, pero además utilizamos el resto de las métricas como colchones que nos ayudan a medir la escala de error.

Estas métricas se aplican a las variables:

- `NA_Sales`
- `EU_Sales`
- `JP_Sales`
- `Other_Sales`
- `Global_Sales`

| Métrica | Significado | Cómo Interpretarla |
|---|---|---|
| `MAE` (Mean Absolute Error) | Calcula el promedio del error absoluto entre los valores reales y las predicciones. | Mientras más cercano a `0`, mejor precisión tiene el modelo. |
| `MSE` (Mean Squared Error) | Calcula el promedio de los errores elevados al cuadrado. Penaliza más los errores grandes. | Valores pequeños indican mejores predicciones. |
| `RMSE` (Root Mean Squared Error) | Es la raíz cuadrada del MSE. Permite interpretar el error en la misma escala de los datos originales. | Mientras menor sea, mejor desempeño tiene el modelo. |
| `R² Score` | Mide qué tan bien el modelo explica la variabilidad de los datos. | Valores cercanos a `1` indican un modelo muy preciso. |

---

### Métricas de Clasificación
Estas métricas fueron escogidas en base a papeles de trabajos de clasificación como lo son **GPT Sniffer** [Ver Paper](https://www.sciencedirect.com/science/article/pii/S0164121224001043?via%3Dihub), un research de un modelo encargado de clasificar fragmentos de código cómo entre hechos por ChatGPT o humanos, un gran punto de referencia para nuestras métricas de clasificación a utilizar.

Estas métricas se aplican a:

- `Critic_Score_Class`

| Métrica | Significado | Cómo Interpretarla |
|---|---|---|
| `Accuracy` | Mide el porcentaje total de predicciones correctas realizadas por el modelo. | Valores cercanos a `1` representan mayor exactitud. |
| `Precision` | Mide cuántas predicciones positivas realizadas por el modelo fueron correctas. | Un valor alto indica pocas falsas predicciones positivas. |
| `Recall` | Mide cuántos casos positivos reales fueron correctamente identificados. | Un valor alto indica que el modelo detecta correctamente la mayoría de los casos reales. |
| `F1 Score` | Combina Precision y Recall en una sola métrica balanceada. | Valores cercanos a `1` representan mejor equilibrio entre precisión y recuperación. |

---

### Métrica Interna del Random Forest

| Métrica | Significado | Cómo Interpretarla |
|---|---|---|
| `OOB Score` (Out Of Bag Score) | Métrica interna del algoritmo Random Forest que evalúa el rendimiento utilizando muestras no vistas durante el entrenamiento. | Valores cercanos a `1` indican que el modelo generaliza correctamente y reduce el riesgo de sobreajuste. |

---

## Resultados de la Versión #1 
Una vez entrenado el modelo tanto con train y test, analizaremos cuáles fueron los resultados que logramos obtener para todas las variables.

### TRAIN
Primero tomemos un vistazo a los valores que esta prediciendo nuestro modelo, podemos notar como en la mayoría de los casos de ventas, los valores predecidos están por encima del valor real, algo que se debe tomar en cuenta para cuando realicemos ajustes. Por otra parte los valores de clasificación en la mayoría de casos parece ser haber predicho correctamente.

<img width="1630" height="696" alt="image" src="https://github.com/user-attachments/assets/a1a8f9ad-a445-4eaf-b578-5ed3f5276c46" />

Ahora vamos a ir variable por variable para analizar como es el rendimiento de cada una:

NA_Sales:
<img width="320" height="264" alt="image" src="https://github.com/user-attachments/assets/18268100-d371-4268-aa99-f791b8e5d066" />

- MAE: 0.2285
- MSE: 0.4825
- RMSE: 0.6947
- R² Score: 0.5002

El modelo obtuvo un desempeño moderado en la predicción de ventas en Norteamérica. El valor de `R² Score = 0.5002` indica que el modelo logra explicar aproximadamente el 50% de la variabilidad de los datos. Además, el `MAE = 0.2285` muestra que el error promedio de predicción es relativamente bajo, aunque todavía existen diferencias considerables entre algunos valores reales y predichos.

EU_Sales:
<img width="320" height="248" alt="image" src="https://github.com/user-attachments/assets/95b11654-544f-41c4-a804-16d953dc39ac" />

- MAE: 0.1562
- MSE: 0.2578
- RMSE: 0.5077
- R² Score: 0.4847

Para las ventas en Europa, el modelo presentó un desempeño aceptable con un `R² Score = 0.4847`, indicando que cerca del 48% de la variabilidad fue explicada correctamente. El `RMSE = 0.5077` refleja que el modelo mantiene errores moderados en las predicciones, mostrando una capacidad razonable para estimar las ventas europeas.

JP_Sales:
<img width="317" height="250" alt="image" src="https://github.com/user-attachments/assets/e4102ca8-81e1-4db7-bdaa-4a267c5cee94" />

- MAE: 0.0468
- MSE: 0.0424
- RMSE: 0.2060
- R² Score: 0.5175

La predicción de ventas en Japón mostró uno de los mejores desempeños del modelo. El `MAE = 0.0468` y el `RMSE = 0.2060` indican errores muy bajos en comparación con las demás variables. Además, el `R² Score = 0.5175` demuestra que el modelo logra explicar más del 51% de la variabilidad de los datos, pero estos valores no podrían ser del todo representativos, ya que en muchos de los datos de Japón el valor es 0, algo que afecta a las predicciones por tanta repetición.

Other_Sales:
<img width="346" height="252" alt="image" src="https://github.com/user-attachments/assets/69eb4591-65fa-47b5-a39a-d21b558c94c6" />

- MAE: 0.0546
- MSE: 0.0491
- RMSE: 0.5077
- R² Score: 0.4847

En las ventas de otras regiones, el modelo obtuvo resultados similares a los de Europa. El `MAE = 0.0546` muestra un error promedio relativamente pequeño, mientras que el `R² Score = 0.4847` indica un desempeño moderado en la capacidad predictiva del modelo.

Global_Sales:
<img width="356" height="253" alt="image" src="https://github.com/user-attachments/assets/49727fc7-3f8f-4a4a-b413-0e6f77e5e426" />

- MAE: 0.4509
- MSE: 2.0345
- RMSE: 1.4264
- R² Score: 0.5042

La predicción de ventas globales presentó errores más altos en comparación con las demás variables. Aunque el `R² Score = 0.5042` indica que el modelo explica aproximadamente el 50% de la variabilidad, el `RMSE = 1.4264` evidencia diferencias más grandes entre las ventas reales y las predicciones realizadas.

Critic_Score_Class:
<img width="414" height="256" alt="image" src="https://github.com/user-attachments/assets/efd7caea-95f5-4e53-b154-77e0954302f9" />

- Accuracy: 0.5486
- Precision: 0.6948
- Recall: 0.5486
- F1 Score: 0.5335

Para la clasificación de críticas, el modelo alcanzó un `Accuracy = 0.5486`, lo que significa que aproximadamente el 55% de las clasificaciones fueron correctas. La `Precision = 0.6948` indica que las predicciones positivas fueron relativamente confiables, mientras que el `F1 Score = 0.5335` refleja un desempeño moderado en el equilibrio entre precisión y recuperación.

Así mismo tomamos un vistazo a nuestra matriz de confusión de nuestro modelo al trabajar con train, esta es una tabla que desglosa el número de instancias reales de una clase específica frente al número de instancias previstas para esa clase.

<img width="859" height="778" alt="image" src="https://github.com/user-attachments/assets/c1766785-a438-4949-bc4a-5df4d345b0a2" />

Podemos ver que la confusión que muestra el desempeño del modelo al clasificar la variable `Critic_Score_Class`, en ella se comparan los valores reales contra las predicciones realizadas por el modelo, permitiendo identificar cuántas clasificaciones fueron correctas y cuántas fueron incorrectas. Los valores ubicados en la diagonal principal representan las predicciones acertadas, mientras que los valores fuera de la diagonal indican confusiones entre clases. 

En general, el modelo presenta un desempeño moderado, logrando identificar correctamente una cantidad importante de ejemplos, aunque todavía existen errores al diferenciar categorías de crítica similares.


### TEST


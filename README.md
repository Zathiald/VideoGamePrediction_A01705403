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

# Métricas de Regresión

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

# Métricas de Clasificación

Estas métricas se aplican a:

- `Critic_Score_Class`

| Métrica | Significado | Cómo Interpretarla |
|---|---|---|
| `Accuracy` | Mide el porcentaje total de predicciones correctas realizadas por el modelo. | Valores cercanos a `1` representan mayor exactitud. |
| `Precision` | Mide cuántas predicciones positivas realizadas por el modelo fueron correctas. | Un valor alto indica pocas falsas predicciones positivas. |
| `Recall` | Mide cuántos casos positivos reales fueron correctamente identificados. | Un valor alto indica que el modelo detecta correctamente la mayoría de los casos reales. |
| `F1 Score` | Combina Precision y Recall en una sola métrica balanceada. | Valores cercanos a `1` representan mejor equilibrio entre precisión y recuperación. |

---

# Métrica Interna del Random Forest

| Métrica | Significado | Cómo Interpretarla |
|---|---|---|
| `OOB Score` (Out Of Bag Score) | Métrica interna del algoritmo Random Forest que evalúa el rendimiento utilizando muestras no vistas durante el entrenamiento. | Valores cercanos a `1` indican que el modelo generaliza correctamente y reduce el riesgo de sobreajuste. |

---

## Resultados de la Versión #1 
Una vez entrenado el modelo tanto con train y test, analizaremos cuáles fueron los resultados que logramos obtener para todas las variables.

# TRAIN
Prime

<img width="1630" height="696" alt="image" src="https://github.com/user-attachments/assets/a1a8f9ad-a445-4eaf-b578-5ed3f5276c46" />




# TEST




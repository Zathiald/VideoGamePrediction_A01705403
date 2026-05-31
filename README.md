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
| 📜 **Video game sales prediction model using regression model** | Papel escrito por **Affana** , **Sourabh Vishwakarmab** y el **Dr. Santosh Kumarc** en la  Universidad Galgotias, en donde a través de User y Critic count, con un modelo de regresión lineal se trata de predecir ventas de videojuegos. | [Ver Paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4935036) |

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

#### Predicciones Generales
Primero tomemos un vistazo a los valores que esta prediciendo nuestro modelo, podemos notar como en la mayoría de los casos de ventas, los valores predecidos están por encima del valor real, algo que se debe tomar en cuenta para cuando realicemos ajustes. Por otra parte los valores de clasificación en la mayoría de casos parece ser haber predicho correctamente.

<img width="1630" height="696" alt="image" src="https://github.com/user-attachments/assets/a1a8f9ad-a445-4eaf-b578-5ed3f5276c46" />

Ahora vamos a ir variable por variable para analizar como es el rendimiento de cada una:

#### NA_Sales:
<img width="320" height="264" alt="image" src="https://github.com/user-attachments/assets/18268100-d371-4268-aa99-f791b8e5d066" />

- MAE: 0.2285
- MSE: 0.4825
- RMSE: 0.6947
- R² Score: 0.5002

El modelo obtuvo un desempeño moderado en la predicción de ventas en Norteamérica. El valor de `R² Score = 0.5002` indica que el modelo logra explicar aproximadamente el 50% de la variabilidad de los datos. Además, el `MAE = 0.2285` muestra que el error promedio de predicción es relativamente bajo, aunque todavía existen diferencias considerables entre algunos valores reales y predichos.

#### EU_Sales:
<img width="320" height="248" alt="image" src="https://github.com/user-attachments/assets/95b11654-544f-41c4-a804-16d953dc39ac" />

- MAE: 0.1562
- MSE: 0.2578
- RMSE: 0.5077
- R² Score: 0.4847

Para las ventas en Europa, el modelo presentó un desempeño aceptable con un `R² Score = 0.4847`, indicando que cerca del 48% de la variabilidad fue explicada correctamente. El `RMSE = 0.5077` refleja que el modelo mantiene errores moderados en las predicciones, mostrando una capacidad razonable para estimar las ventas europeas.

#### JP_Sales:
<img width="317" height="250" alt="image" src="https://github.com/user-attachments/assets/e4102ca8-81e1-4db7-bdaa-4a267c5cee94" />

- MAE: 0.0468
- MSE: 0.0424
- RMSE: 0.2060
- R² Score: 0.5175

La predicción de ventas en Japón mostró uno de los mejores desempeños del modelo. El `MAE = 0.0468` y el `RMSE = 0.2060` indican errores muy bajos en comparación con las demás variables. Además, el `R² Score = 0.5175` demuestra que el modelo logra explicar más del 51% de la variabilidad de los datos, pero estos valores no podrían ser del todo representativos, ya que en muchos de los datos de Japón el valor es 0, algo que afecta a las predicciones por tanta repetición.

#### Other_Sales:
<img width="346" height="252" alt="image" src="https://github.com/user-attachments/assets/69eb4591-65fa-47b5-a39a-d21b558c94c6" />

- MAE: 0.0546
- MSE: 0.0491
- RMSE: 0.5077
- R² Score: 0.4847

En las ventas de otras regiones, el modelo obtuvo resultados similares a los de Europa. El `MAE = 0.0546` muestra un error promedio relativamente pequeño, mientras que el `R² Score = 0.4847` indica un desempeño moderado en la capacidad predictiva del modelo.

#### Global_Sales:
<img width="356" height="253" alt="image" src="https://github.com/user-attachments/assets/49727fc7-3f8f-4a4a-b413-0e6f77e5e426" />

- MAE: 0.4509
- MSE: 2.0345
- RMSE: 1.4264
- R² Score: 0.5042

La predicción de ventas globales presentó errores más altos en comparación con las demás variables. Aunque el `R² Score = 0.5042` indica que el modelo explica aproximadamente el 50% de la variabilidad, el `RMSE = 1.4264` evidencia diferencias más grandes entre las ventas reales y las predicciones realizadas.

#### Critic_Score_Class:
<img width="414" height="256" alt="image" src="https://github.com/user-attachments/assets/efd7caea-95f5-4e53-b154-77e0954302f9" />

- Accuracy: 0.5486
- Precision: 0.6948
- Recall: 0.5486
- F1 Score: 0.5335

Para la clasificación de críticas, el modelo alcanzó un `Accuracy = 0.5486`, lo que significa que aproximadamente el 55% de las clasificaciones fueron correctas. La `Precision = 0.6948` indica que las predicciones positivas fueron relativamente confiables, mientras que el `F1 Score = 0.5335` refleja un desempeño moderado en el equilibrio entre precisión y recuperación.

#### Matriz de Confusión 

Así mismo tomamos un vistazo a nuestra matriz de confusión de nuestro modelo al trabajar con train, esta es una tabla que desglosa el número de instancias reales de una clase específica frente al número de instancias previstas para esa clase.

<img width="859" height="778" alt="image" src="https://github.com/user-attachments/assets/c1766785-a438-4949-bc4a-5df4d345b0a2" />

Podemos ver que la confusión que muestra el desempeño del modelo al clasificar la variable `Critic_Score_Class`, en ella se comparan los valores reales contra las predicciones realizadas por el modelo, permitiendo identificar cuántas clasificaciones fueron correctas y cuántas fueron incorrectas. Los valores ubicados en la diagonal principal representan las predicciones acertadas, mientras que los valores fuera de la diagonal indican confusiones entre clases. 

En general, el modelo presenta un desempeño moderado, logrando identificar correctamente una cantidad importante de ejemplos, aunque todavía existen errores al diferenciar categorías de crítica similares.


### TEST

#### Predicciones Generales
Ahora tomemos un vistazo a los valores que esta prediciendo nuestro modelo una vez que le damos el dataset de test, podemos notar que en los datos numéricos hay un mucho mayor número de datos que predicen por abajo del valor real y además hay muchos más casos en donde la clasificación falla.

<img width="1731" height="698" alt="image" src="https://github.com/user-attachments/assets/66ab7172-a210-4e24-b401-a1b02ad8d75d" />


Ahora vamos a ir variable por variable para analizar como es el rendimiento de cada una:

#### NA_Sales:
<img width="329" height="255" alt="image" src="https://github.com/user-attachments/assets/73b54778-011e-48f6-8ac5-547254c2ab85" />

- MAE: 0.3246
- MSE: 0.6393
- RMSE: 0.7995
- R² Score: 0.0712

En las ventas de Norteamérica, el modelo presentó una disminución importante en el desempeño respecto a los resultados anteriores. El `R² Score = 0.0712` indica que el modelo apenas logra explicar una pequeña parte de la variabilidad de los datos. Además, el incremento en `MAE` y `RMSE` muestra que las predicciones tienen un error considerablemente mayor en comparación con el entrenamiento.

#### EU_Sales:
<img width="327" height="258" alt="image" src="https://github.com/user-attachments/assets/1c19210d-7055-4f2f-9079-8c98021e7203" />

- MAE: 0.2372
- MSE: 0.3247
- RMSE: 0.5698
- R² Score: -0.0809

Para las ventas en Europa, el desempeño empeoró notablemente en comparación con los resultados anteriores. El `R² Score = -0.0809` indica que el modelo no logró generalizar correctamente sobre los datos de prueba. Aunque el `RMSE = 0.5698` sigue siendo moderado, las predicciones presentan más diferencias respecto a los valores reales.


#### JP_Sales:
<img width="322" height="251" alt="image" src="https://github.com/user-attachments/assets/1d27d69f-f636-47b9-b1d5-9ec25ea5f38e" />

- MAE: 0.0629
- MSE: 0.0452
- RMSE: 0.2127
- R² Score: 0.0958

La variable `JP_Sales` continúa siendo una de las mejores predicciones del modelo. Aunque el `R² Score = 0.0958` es menor al obtenido anteriormente, los errores `MAE` y `RMSE` permanecen relativamente bajos, indicando que el modelo mantiene cierta estabilidad en las ventas de Japón, lo cual aún no es representativo por la repetición del 0 en muchas de las instancias.


#### Other_Sales:
<img width="351" height="260" alt="image" src="https://github.com/user-attachments/assets/66eedafd-0afc-4651-be12-4c9e71a540a7" />

- MAE: 0.0764
- MSE: 0.0408
- RMSE: 0.2019
- R² Score: -0.0583

En las ventas de otras regiones, el modelo mostró un rendimiento bajo durante la evaluación. El `R² Score = -0.0583` evidencia que el modelo no logra explicar correctamente la variabilidad de los datos de prueba. Sin embargo, los errores absolutos siguen siendo relativamente pequeños debido a que los valores de esta variable suelen ser bajos.

#### Global_Sales:
<img width="361" height="265" alt="image" src="https://github.com/user-attachments/assets/86bae307-99c8-43fd-b139-adaad29a6f76" />

- MAE: 0.6505
- MSE: 2.3818
- RMSE: 1.5433
- R² Score: -0.0189

La predicción de ventas globales presentó nuevamente los errores más altos entre todas las variables. El `RMSE = 1.5433` y el `MAE = 0.6505` muestran diferencias considerables entre valores reales y predicciones. Además, el `R² Score = -0.0189` indica que el modelo tiene dificultades para generalizar correctamente las ventas globales en datos no vistos.

#### Critic_Score_Class:
<img width="420" height="259" alt="image" src="https://github.com/user-attachments/assets/eaad40d9-d022-4fcb-b580-49941024dac0" />

- Accuracy: 0.3443
- Precision: 0.3540
- Recall: 0.3443
- F1 Score: 0.2965

La clasificación de críticas mostró una disminución importante respecto al entrenamiento. El `Accuracy = 0.3443` indica que el modelo solo logró clasificar correctamente aproximadamente el 34% de los casos. De igual manera, `Precision`, `Recall` y `F1 Score` disminuyeron considerablemente, lo que sugiere que el modelo presenta problemas de generalización y posibles indicios de sobreajuste.


#### Matriz de Confusión 

Y revisaremos nuevamente que tanto cambio la matriz de confusión para revisar el desempeño del dataset de test.

<img width="792" height="730" alt="image" src="https://github.com/user-attachments/assets/2bbf0f84-ee4c-4627-8c98-23b61f0a7306" />

La matriz de confusión del conjunto de prueba muestra que el modelo tuvo dificultades para clasificar correctamente las categorías de `Critic_Score_Class` en datos no vistos. Aunque existen algunas predicciones correctas en la diagonal principal, también se observan numerosas confusiones entre las clases `1` y `2`, así como entre las clases `0` y `1`. Esto indica que el modelo no logra diferenciar claramente algunas categorías de crítica. En comparación con la matriz de entrenamiento, el desempeño disminuyó considerablemente.

#### Conclusiones Version #1

En general, el modelo presentó un desempeño moderado en la predicción de las variables de ventas y de la clasificación `Critic_Score_Class`, durante el entrenamiento, el modelo logró obtener métricas relativamente buenas, especialmente en algunas variables como `JP_Sales`, pero esto podria ser debido a la repeticion del 0 en JP_Sales.

Sin embargo, al evaluar el modelo con datos de prueba, las métricas disminuyeron considerablemente, mostrando problemas de generalización. Esto sugiere la presencia de **overfitting**, ya que el modelo se adaptó demasiado a los datos de entrenamiento y perdió precisión al trabajar con datos no vistos, pero a pesar de ello, el modelo logró identificar ciertas tendencias relevantes y sirve como una base para los cambios y ajustes próximos que se van a realizar.

## Cambios de la Versión #2

### Separación de Modelos
Uno de los puntos a destacar es el hecho de que en la primera versión del modelo, tanto la regresión como la clasificación se encontraban en un solo modelo, esto hacía que los datos se mezclaran y se volvieran unos difíciles de predecir y otros de clasificar, además que le proporcionaba muy pocos parámetros al modelo para poder realizar las clasificaciones debidas, por ello el mayor cambio que se puede realizar es el dividir el modelo en dos, uno para clasificación y otro para predicción.

Para esto lo primero que se hizo fue separar los datos de predicción y de input para nuestro modelo de regresión y de clasificación, para clasificación nuestros valores quedan iguales:

```python
X = vg_data[["Platform", "Genre", "Publisher", "Rating"]]

y_reg = vg_data[
    [
        "NA_Sales",
        "EU_Sales",
        "JP_Sales",
        "Other_Sales",
        "Global_Sales"
    ]
]
```
Pero ahora para clasificación lo que hicimos fue incluir las ventas como parámetros, esto con el fin de tener más valores reales representativos e importantes que ayuden al modelo realizar la clasificación correcta.

```python
X_class = vg_data[[
    "Platform",
    "Genre",
    "Publisher",
    "Rating",
    "NA_Sales",
    "EU_Sales",
    "JP_Sales",
    "Other_Sales",
    "Global_Sales"
]]

y_class = vg_data["Critic_Score_Class"]
```

Ahora que tenemos la división de variables, tenemos que realizar la codificación y configuración de las variables tanto para regresión y clasificación, un punto importante es que debido al uso de un **MLP Regressor** para regresión, vamos a escalar las variables parámetro de regresión, con el fin de tener datos más centrados para el modelo:

```python
categorical_features = [
    "Platform",
    "Genre",
    "Publisher",
    "Rating"
]

ct = ColumnTransformer(
    transformers=[
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)

scaler_x = MaxAbsScaler()

X = ct.fit_transform(X)

X = scaler_x.fit_transform(X)
```

Y por otra parte hacemos la codificación para los parámetros de clasificación:

```python
categorical_features_class = [
    "Platform",
    "Genre",
    "Publisher",
    "Rating"
]

numeric_features_class = [
    "NA_Sales",
    "EU_Sales",
    "JP_Sales",
    "Other_Sales",
    "Global_Sales"
]

ct_class = ColumnTransformer(

    transformers=[

        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features_class
        ),

        (
            "num",
            StandardScaler(),
            numeric_features_class
        )
    ]
)

X_class = ct_class.fit_transform(X_class)
```

Otro punto importante a notar es que al momento de realizar clasificación, el dato de Publisher causa mucho ruido por la variedad de datos, los cuales la mayoría se repiten tan solo una vez, entonces para ello primero haremos el split de datos para regresión:

```python
X_train, X_test, y_train_reg, y_test_reg = train_test_split(
    X,
    y_reg,
    test_size=0.2,
    random_state=42,
    stratify=y_class
)
```

Luego agrupamos los datos de publisher que se repitan muy poco en un solo valor llamado Other, para evitar la mayor cantidad de ruido en la clasificación y después de la agrupación ahora dividimos el dataset para clasificación:

```python
top_publishers = vg_data["Publisher"].value_counts().nlargest(20).index

vg_data["Publisher"] = vg_data["Publisher"].apply(

    lambda x: x if x in top_publishers else "Other"
)

X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(

    X_class,
    y_class,

    test_size=0.2,

    random_state=42,

    stratify=y_class
)
```

Ahora que hemos preparado nuevamente los datos, vamos a crear los dos modelos diferentes, iniciamos con el de regresión, para el cual usaremos un MLP Regressor, un MLP es un modelo de red neuronal artificial diseñado para predecir valores numéricos continuos, justo lo que buscamos lograr para la predicción de ventas, de igual manera este modelo fue utilizado en el paper escrito por la universidad de Galgotias, dándoles el segundo mejor resultado.

Entonces para esto vamos a tomar los siguientes parámetros al crear el modelo:


| Parámetro | Propósito | Valor utilizado | Justificación |
|------------|------------|----------------|---------------|
| `hidden_layer_sizes` | Define el número de capas ocultas y neuronas por capa. | `(32, 16)` | Se utilizaron dos capas ocultas decrecientes para capturar relaciones no lineales entre las características del videojuego y las ventas, manteniendo una complejidad moderada que reduzca el riesgo de sobreajuste. |
| `activation` | Función de activación utilizada en las capas ocultas. | `relu` | ReLU mejora la eficiencia computacional y favorece la convergencia del entrenamiento, especialmente en redes neuronales con múltiples capas. |
| `solver` | Algoritmo de optimización para actualizar los pesos. | `adam` | Adam es uno de los optimizadores más utilizados para redes neuronales debido a su rapidez de convergencia y buen desempeño en conjuntos de datos con características dispersas provenientes de One-Hot Encoding. |
| `alpha` | Factor de regularización L2. | `0.1` | Se incorporó regularización para limitar la magnitud de los pesos y reducir el sobreajuste, considerando el elevado número de variables generadas por la codificación categórica. |
| `learning_rate_init` | Tasa de aprendizaje inicial. | `0.0005` | Se seleccionó una tasa de aprendizaje conservadora para lograr una convergencia más estable y evitar oscilaciones durante la optimización. |
| `max_iter` | Número máximo de iteraciones de entrenamiento. | `1000` | Se permitió un número elevado de iteraciones para asegurar que el modelo tuviera suficiente tiempo para converger. |
| `early_stopping` | Detención automática cuando el desempeño deja de mejorar. | `True` | Evita continuar entrenando cuando la mejora en el conjunto de validación es mínima, ayudando a prevenir sobreajuste. |
| `validation_fraction` | Porcentaje del conjunto de entrenamiento reservado para validación. | `0.1` | Se utilizó el 10% de los datos de entrenamiento para monitorear el rendimiento del modelo durante el aprendizaje. |
| `n_iter_no_change` | Número de iteraciones sin mejora antes de detener el entrenamiento. | `20` | Permite cierta tolerancia a pequeñas fluctuaciones antes de considerar que el modelo ha convergido. |
| `random_state` | Semilla para la generación aleatoria. | `42` | Garantiza la reproducibilidad de los resultados obtenidos durante el entrenamiento y la evaluación. |

Ahora con estos valores programamos el modelo y lo entrenamos:

```python
regressor = MLPRegressor(

    hidden_layer_sizes=(32, 16),

    activation="relu",

    solver="adam",

    alpha=0.1,

    learning_rate_init=0.0005,

    max_iter=1000,

    early_stopping=True,

    validation_fraction=0.1,

    n_iter_no_change=20,

    random_state=42,
)

regressor.fit(
    X_train,
    y_train_reg_scaled
)
```

Después, vamos a realizar el modelo para la clasificación, utilizaremos RandomForest aún pero utilizaremos la versión de Classifier, pero para este modelo haremos algo diferente, aquí lo que haremos es utilizar un random search que va probando diferentes parámetros con el fin de obtener el mejor resultado:

```python
rf = RandomForestClassifier(

    bootstrap=True,

    oob_score=True,

    random_state=42,

    n_jobs=-1
)

param_distributions = {

    "n_estimators": [100, 200, 300, 500],

    "max_depth": [10, 20, 30, None],

    "min_samples_split": [2, 5, 10],

    "min_samples_leaf": [1, 2, 5],

    "max_features": ["sqrt", "log2"],

    "criterion": ["gini", "entropy"]
}

random_search = RandomizedSearchCV(

    estimator=rf,

    param_distributions=param_distributions,

    n_iter=20,

    cv=5,

    scoring="f1_weighted",

    verbose=2,

    random_state=42,

    n_jobs=-1
)

random_search.fit(

    X_train_class,

    y_train_class
)
```

Los resultados fueron los siguientes:

| Parámetro | Valor Seleccionado | Descripción |
|------------|-------------------|-------------|
| `n_estimators` | 500 | Número de árboles del bosque. Un mayor número de árboles suele producir predicciones más estables. |
| `min_samples_split` | 5 | Cantidad mínima de muestras necesarias para dividir un nodo. |
| `min_samples_leaf` | 1 | Número mínimo de muestras requerido en una hoja terminal. |
| `max_features` | `log2` | Cantidad de variables consideradas en cada división del árbol. |
| `max_depth` | `None` | Permite que los árboles crezcan sin límite de profundidad. |
| `criterion` | `gini` | Medida utilizada para evaluar la calidad de cada división. |

### Resultados de la Versión #2

El primer valor que obtuvimos fue el OOB Score del RandomForest Classifier

<img width="163" height="121" alt="image" src="https://github.com/user-attachments/assets/a36800c1-b4dc-443e-9d02-8f4a203b39c5" />

El mejor modelo obtuvo un **OOB-Score ponderado de 0.4651** durante la validación cruzada, y el **OOB Score (Out Of Bag Score)** del modelo final fue de **0.4713**, lo que indica que el rendimiento estimado utilizando muestras no vistas durante el entrenamiento es muy similar al obtenido en validación cruzada, esta cercanía entre ambas métricas sugiere que el modelo presenta una capacidad de generalización consistente sin señales de sobreajuste.

### TRAIN

#### Predicciones Generales
Primero tomemos un vistazo a los valores que esta prediciendo nuestro modelo, podemos notar que en la clasificación hay un muchísimo mejor resultado en comparación con el modelo anterior y así mismo en la predicción, aunque aún en los valores más altos realiza algunas confusiones.

<img width="1176" height="484" alt="image" src="https://github.com/user-attachments/assets/6f705ef4-0221-44f9-815b-23fc7e041109" />

Ahora vamos a ir variable por variable para analizar como es el rendimiento de cada una:

#### NA_Sales:
<img width="263" height="169" alt="image" src="https://github.com/user-attachments/assets/a46a3ec0-f6b5-4d9e-9fe1-c6ba860d6db3" />

- MAE: 0.2727
- MSE: 0.6777
- RMSE: 0.8232
- R² Score: 0.3151

El modelo obtuvo un `R² Score = 0.3151`, lo que indica que explica aproximadamente el 31% de la variabilidad de las ventas en Norteamérica y aunque el desempeño en entrenamiento es inferior al de la Versión #1 (`R² = 0.5002`), esta reducción es esperada debido a la incorporación de regularización y técnicas para disminuir el sobreajuste, reflejando un comportamiento más conservador del modelo.

#### EU_Sales:
<img width="257" height="170" alt="image" src="https://github.com/user-attachments/assets/d5c4c692-0144-4c5a-97e6-15f946e0790c" />

- MAE: 0.1855
- MSE: 0.3447
- RMSE: 0.5871
- R² Score: 0.2619

Para las ventas en Europa, el modelo alcanzó un `R² Score = 0.2619`, si bien este resultado es menor al obtenido en la Versión #1 (`R² = 0.4847`), el modelo presenta una menor tendencia a memorizar los datos de entrenamiento, esto sugiere que la nueva arquitectura prioriza una mejor capacidad de generalización sobre el ajuste perfecto de los datos de entrenamiento, lo cuál es ideal ya que buscamos que aprenda, no que memorice.

#### JP_Sales:
<img width="260" height="170" alt="image" src="https://github.com/user-attachments/assets/361870cd-8ff0-4051-adad-28aeecaddd9b" />

- MAE: 0.0628
- MSE: 0.0506
- RMSE: 0.2249
- R² Score: 0.3481

La variable `JP_Sales` continúa mostrando uno de los mejores desempeños relativos dentro de las variables de regresión, aunque el `R² Score = 0.3481` es menor al de la Versión #1 (`R² = 0.5175`), los errores siguen siendo relativamente bajos, sin embargo, sigue siendo importante considerar que una gran cantidad de registros presentan valores cercanos a cero, lo que puede influir positivamente en estas métricas.

#### Other_Sales:
<img width="277" height="168" alt="image" src="https://github.com/user-attachments/assets/940dff3f-7102-4c77-808d-9b90c7311346" />

- MAE: 0.0615
- MSE: 0.0496
- RMSE: 0.2227
- R² Score: 0.2260

Para las ventas en otras regiones, el modelo obtuvo un `R² Score = 0.2260`. Comparado con la Versión #1 (`R² = 0.4847`), existe una disminución en el ajuste sobre los datos de entrenamiento, aun así, el error absoluto continúa siendo reducido debido a la baja magnitud que suelen presentar estas ventas en comparación con otras regiones.

#### Global_Sales:
<img width="279" height="166" alt="image" src="https://github.com/user-attachments/assets/d4864ed1-8550-4d38-8117-e911e548ef29" />

- MAE: 0.5110
- MSE: 2.5339
- RMSE: 1.5918
- R² Score: 0.3453

La predicción de ventas globales alcanzó un `R² Score = 0.3453` y aunque este resultado es menor que el obtenido en la Versión #1 (`R² = 0.5042`), esto refleja un modelo menos propenso al sobreajuste, los errores continúan siendo los más altos entre todas las variables debido a que esta métrica concentra la variabilidad de todas las regiones.

#### Critic_Score_Class:
<img width="321" height="170" alt="image" src="https://github.com/user-attachments/assets/f8e5ffdb-6159-44b1-b60d-448006ff7393" />

- Accuracy: 0.9597
- Precision: 0.9601
- Recall: 0.9597
- F1 Score: 0.9598

La clasificación de críticas mostró una mejora muy importante durante el entrenamiento. El modelo alcanzó un `Accuracy = 0.9597` y un `F1 Score = 0.9598`, superando ampliamente los resultados de la Versión #1 (`Accuracy = 0.5486`, `F1 = 0.5335`), esto demuestra que la incorporación de las variables de ventas como parámetros de entrada aporta información valiosa para la clasificación de la crítica.

#### Matriz de Confusión 

Así mismo tomamos un vistazo a nuevamente a la matriz de confusión 

<img width="569" height="518" alt="image" src="https://github.com/user-attachments/assets/f3db5800-de07-4c45-b3c7-a7a5dfb91b6f" />

Podemos ver como el modelo ha mejorado bastante en clasificar las categorías correctamente, con un color diagonal directo que representa la mayoría de los datos clasificados correctamente.


### TEST

#### Predicciones Generales
Tomamos nuevamente un vistazo a los valores que esta prediciendo nuestro modelo, podemos notar una gran mejora en cuánto a la clasificación y predicción de valores en comparación con el antiguo modelo, aún existe confusión en ventas altas, pero es menor que la versión #1

<img width="1206" height="482" alt="image" src="https://github.com/user-attachments/assets/dc2ddecc-076c-4393-87ee-9138ac36eaf8" />

Ahora vamos a ir variable por variable para analizar como es el rendimiento de cada una:

#### NA_Sales:
<img width="254" height="174" alt="image" src="https://github.com/user-attachments/assets/82cc3b97-5751-47d7-a5df-9ed803a41561" />

- MAE: 0.3013
- MSE: 0.4872
- RMSE: 0.6947
- R² Score: 0.1841

En el conjunto de prueba, el modelo mejoró respecto a la Versión #1. El `R² Score` aumentó de `0.0712` a `0.1841`, mientras que el `MAE` disminuyó de `0.3246` a `0.3013`, esto indica que el modelo logra generalizar mejor las ventas en Norteamérica y reducir el error promedio de predicción.

#### EU_Sales:
<img width="250" height="174" alt="image" src="https://github.com/user-attachments/assets/3d4d55cd-7988-4d5a-8845-e709aa53f8ac" />

- MAE: 0.2229
- MSE: 0.3823
- RMSE: 0.6183
- R² Score: 0.1383

Las ventas en Europa mostraron una mejora considerable frente a la Versión #1. El `R² Score` pasó de `-0.0809` a `0.1383`, lo que significa que el modelo ahora logra capturar parte de la variabilidad presente en los datos de prueba, y aunque aún existe margen de mejora, la capacidad de generalización aumentó significativamente.

#### JP_Sales:
<img width="248" height="173" alt="image" src="https://github.com/user-attachments/assets/e1d1495c-e4d1-4fd4-9be8-c79a9167b4c5" />

- MAE: 0.072
- MSE: 0.0716
- RMSE: 0.2675
- R² Score: 0.2160

La variable `JP_Sales` presentó una mejora notable en generalización. El `R² Score` aumentó de `0.0958` a `0.2160`, mostrando que el modelo logra explicar una mayor proporción de la variabilidad de los datos de prueba, no obstante, sigue siendo necesario considerar la alta presencia de valores cercanos a cero dentro del dataset, lo que sugiere que se debería buscar más datos representativos.

#### Other_Sales:
<img width="271" height="172" alt="image" src="https://github.com/user-attachments/assets/2aa05382-cb2b-4443-a385-56717a69635d" />

- MAE: 0.0809
- MSE: 0.0870
- RMSE: 0.2949
- R² Score: 0.1043

Las ventas de otras regiones también mejoraron respecto a la versión anterior. El `R² Score` pasó de `-0.0583` a `0.1043`, permitiendo que el modelo explique parte de la variabilidad de los datos de prueba, y aunque los errores aumentaron ligeramente, la mejora en capacidad predictiva es evidente, un gran paso para la mejora del modelo.

#### Global_Sales:
<img width="275" height="174" alt="image" src="https://github.com/user-attachments/assets/44b7a3f7-4b87-4cdb-930e-68ebcb7ea231" />

- MAE: 0.6028
- MSE: 2.8163
- RMSE: 1.6782
- R² Score: 0.1387

La predicción de ventas globales mostró una mejora importante en comparación con la Versión #1. El `R² Score` aumentó de `-0.0189` a `0.1387`, indicando una mejor capacidad para generalizar sobre datos no vistos, además, el `MAE` disminuyó de `0.6505` a `0.6028`, reflejando una reducción en el error promedio.

#### Critic_Score_Class:
<img width="311" height="164" alt="image" src="https://github.com/user-attachments/assets/243cdf52-4840-4ee8-b6c6-e15dbb888798" />

- Accuracy: 0.4631
- Precision: 0.4564
- Recall: 0.4631
- F1 Score: 0.4581


La clasificación de críticas mejoró significativamente en el conjunto de prueba. El `Accuracy` aumentó de `0.3443` a `0.4631`, mientras que el `F1 Score` pasó de `0.2965` a `0.4581`, y aunque el rendimiento aún está lejos de los resultados observados en entrenamiento, la mejora demuestra que la incorporación de información de ventas y la optimización de hiperparámetros permitieron obtener un clasificador más robusto y con mejor capacidad de generalización.

#### Matriz de Confusión 

Así mismo tomamos un vistazo a nuevamente a la matriz de confusión 

<img width="564" height="521" alt="image" src="https://github.com/user-attachments/assets/2a11bb81-79d4-4f99-a7f3-d4ad4ed994bb" />

Se puede notar menor confusión en el modelo que en la versión anterior, con la mayoría de datos apareciendo en sus lugares correctos, pero aún existen gran parte de datos que son confundidos a la hora de ser clasificados correctamente.

### Conclusiones Versión #2

La segunda versión del modelo logró mejoras importantes en la capacidad de generalización respecto a la Versión #1. Aunque las métricas de entrenamiento para las variables de regresión disminuyeron, las métricas de prueba mejoraron en todas las regiones analizadas, lo que indica una reducción del sobreajuste observado anteriormente.

La mejora más significativa se produjo en la clasificación de `Critic_Score_Class`, donde la incorporación de las ventas como variables de entrada y la optimización mediante `RandomizedSearchCV` permitieron incrementar considerablemente las métricas de Accuracy y F1 Score tanto en entrenamiento como en prueba.

A pesar de estos avances, las métricas de regresión continúan mostrando valores de `R²` relativamente bajos, especialmente para las ventas globales, lo que sugiere que aún existen variables relevantes que no están siendo consideradas por el modelo y que podrían incorporarse en futuras versiones para mejorar la precisión de las predicciones, así como a buscar experimentar con otro dataset que incluya una mayor cantidad de datos verídicos.




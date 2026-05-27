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

- NA_Sales
- EU_Sales
- JP_Sales
- Other_Sales
- Global_Sales
- Critic_Score_Class

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



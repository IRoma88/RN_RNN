# 📊 Análisis de Sentimientos con RNNs en Reviews de Películas (IMDB)

Este proyecto aplica **redes neuronales recurrentes (RNN)** para el análisis de sentimientos en reviews de películas del dataset IMDB, disponible desde `keras.datasets`. Se implementan tres variantes: `SimpleRNN`, `LSTM` y `GRU`, evaluando su desempeño en clasificación binaria (positiva o negativa).

## 📁 Estructura del Proyecto

- Carga del dataset IMDB con las 10,000 palabras más frecuentes.
- Preprocesamiento: padding de secuencias a longitud fija.
- Construcción de modelos secuenciales con capas:
  - `Embedding`
  - `SimpleRNN` / `LSTM` / `GRU`
  - `Dense` con activación `sigmoid`
- Entrenamiento con `validation_split=0.2`
- Evaluación con métricas de precisión (`accuracy`)
- Visualización con `matplotlib`

## 🧠 Modelos Entrenados

- Modelo con `SimpleRNN`
- Modelo con `LSTM`
- Modelo con `GRU`

Todos los modelos están construidos sobre una representación vectorial aprendida (`Embedding`) y diseñados para clasificación binaria.

## 🔧 Requisitos

Instala las dependencias con:

```bash
pip install -r requirements.txt
````

## ▶️ Ejecución
Puedes ejecutar el archivo principal en Jupyter Notebook o como script de Python:

````bash
python imdb_sentiment_analysis.py
````

## 📊 Visualización
Se generan gráficas de evolución de precisión y pérdida para entrenamiento y validación por cada arquitectura.

## 📈 Resultados
Se muestran métricas de precisión (accuracy) y pérdida (loss) en el conjunto de test para comparar el rendimiento entre SimpleRNN, LSTM y GRU.

## 📚 Referencias
  . Dataset original: IMDB Sentiment Dataset - Stanford

  . Documentación Keras: https://keras.io/api/datasets/imdb/

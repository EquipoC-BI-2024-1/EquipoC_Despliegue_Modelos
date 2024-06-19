import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_regression, SelectKBest, f_regression
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import datetime

# Función para cargar y limpiar datos
def clean_data(data):
    data = data.dropna()
    return data

# Función para normalizar datos
def normalize_data(data):
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    return pd.DataFrame(data_scaled, columns=data.columns), scaler

# Función para seleccionar las mejores características
def select_features(X, y, num_features):
    mutual_info = mutual_info_regression(X, y)
    k_best = SelectKBest(score_func=f_regression, k=num_features).fit(X, y)
    features = X.columns[k_best.get_support(indices=True)]
    return features.tolist()

# Función para entrenar el modelo LSTM
def train_lstm(X_train, y_train, input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=500, batch_size=32, validation_split=0.2)
    return model

# Función para mostrar la página de LSTM
def mostrar_pagina_lstm():
    st.title("Modelo LSTM")
    st.write("""
        La siguiente implementación es una aplicación para predecir el precio futuro de un instrumento financiero utilizando un modelo de aprendizaje automático LSTM (Long Short-Term Memory).
    """)

    # Entrada de usuario para el ticker del instrumento financiero
    ticker = st.text_input("Ticker del instrumento financiero", value='FSM')
    # Entrada de usuario para las fechas de inicio y fin
    start_date = st.date_input("Fecha de inicio", value=pd.to_datetime('2021-01-01'))
    end_date = st.date_input("Fecha de fin", value=pd.to_datetime('2021-08-11'))
    target_column_name = 'Close'

    # Convertir las fechas a timestamps
    start_date_timestamp = int(datetime.datetime.combine(start_date, datetime.datetime.min.time()).timestamp())
    end_date_timestamp = int(datetime.datetime.combine(end_date, datetime.datetime.min.time()).timestamp())

    # Cargar datos
    url = f'https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={start_date_timestamp}&period2={end_date_timestamp}&interval=1d&events=history&includeAdjustedClose=true'
    data = pd.read_csv(url)

    # Mantener la columna de fechas para las gráficas
    dates = data['Date']
    data = data.drop(columns=['Date'])

    # Limpiar y normalizar los datos
    data = clean_data(data)
    data, scaler = normalize_data(data)

    # Seleccionar las mejores características
    target_column = target_column_name
    num_features = 5  # Número de características a seleccionar
    selected_features = select_features(data.drop(columns=[target_column]), data[target_column], num_features)
    selected_features.append(target_column)
    data = data[selected_features]

    # Separar características y objetivo
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Dividir los datos en conjuntos de entrenamiento y prueba
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    dates_train, dates_test = dates[:train_size], dates[train_size:]

    st.write(f'### Características seleccionadas: {selected_features}')

    # Los datos de entrenamiento y prueba se reestructuran en un formato 3D requerido por LSTM (samples, timesteps, features)
    X_train_lstm = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test_lstm = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))

    # Entrenar el modelo LSTM
    st.write("### Entrenamiento del modelo LSTM")
    st.write("Entrenando el modelo LSTM con los datos históricos seleccionados. Este proceso puede tardar varios minutos.")
    lstm_model = train_lstm(X_train_lstm, y_train, (1, X_train.shape[1]))

    # Generar predicciones sobre los datos de prueba
    st.write("### Generando predicciones")
    st.write("Usando el modelo LSTM entrenado para predecir los precios futuros del instrumento financiero.")
    lstm_predictions = pd.Series(lstm_model.predict(X_test_lstm).flatten(), index=X_test.index)

    # Calcular MAPE y RMSE
    mape_lstm = mean_absolute_percentage_error(y_test, lstm_predictions)
    rmse_lstm = np.sqrt(mean_squared_error(y_test, lstm_predictions))

    # Mostrar las métricas de validación
    st.write(f'### Métricas de validación')
    st.write(f'MAPE (Mean Absolute Percentage Error) del modelo LSTM: {mape_lstm}')
    st.write(f'RMSE (Root Mean Squared Error) del modelo LSTM: {rmse_lstm}')

    # Mostrar predicciones numéricas en un DataFrame
    st.write("### Predicciones numéricas")
    st.write("""
    En esta tabla, se muestran las predicciones generadas por el modelo junto con los valores reales. 
    Esto permite comparar directamente cuánto difieren las predicciones de los valores reales.
    """)
    predictions_df = pd.DataFrame({'Actual': y_test, 'Predicted': lstm_predictions})
    st.dataframe(predictions_df)

    # Graficar las predicciones junto con los valores reales
    st.write("### Gráfico de Predicciones")
    st.write("""
    El siguiente gráfico muestra los valores reales y las predicciones realizadas por el modelo. 
    Este gráfico ayuda a visualizar el desempeño del modelo y ver cómo de cerca las predicciones siguen la tendencia de los datos reales.
    """)
    plt.figure(figsize=(10, 5))
    plt.plot(y_test.values, label='Actual')
    plt.plot(lstm_predictions.values, label='Predicted')
    plt.legend()
    st.pyplot(plt)

    # Predicción del precio para mañana
    st.write("### Predicción de Precio para Mañana")
    last_data_point = X_test_lstm[-1]  # Último punto de datos de prueba en formato LSTM
    prediction_for_tomorrow = lstm_model.predict(np.array([last_data_point]))

    # Invertir la normalización manualmente
    min_close = scaler.data_min_[-1]  # Min de la columna Close
    scale_close = scaler.data_range_[-1]  # Rango de la columna Close

    prediction_for_tomorrow_scaled = prediction_for_tomorrow * scale_close + min_close

    st.write(f'El precio predicho para mañana es: {prediction_for_tomorrow_scaled[0, 0]:.2f}')

    # Recomendación basada en los resultados
    st.write("### Recomendación")
    st.write("""
    - **Realizar un análisis detallado de las características:** Revisar la relevancia y correlación de las características seleccionadas. Considerar la transformación de características para mejorar su representación.
    - **Optimizar los hiperparámetros del modelo LSTM:** Buscar la mejor configuración de capas, neuronas, optimizador y función de pérdida mediante técnicas como la validación cruzada.
    - **Investigar las fuentes de error:** Analizar los errores del modelo para identificar patrones o sesgos. Implementar técnicas de ensemble learning para mejorar la robustez.
    - **Incorporar información adicional:** Considerar indicadores técnicos o análisis de sentimiento del mercado para complementar el modelo.
    - **Monitorear el desempeño:** Evaluar el desempeño del modelo en diferentes subconjuntos de datos y en el tiempo. Realizar ajustes cuando sea necesario.
    """)

if __name__ == "__main__":
    mostrar_pagina_lstm()

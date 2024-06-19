import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.feature_selection import mutual_info_regression, SelectKBest, f_regression
from sklearn.svm import SVR
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import datetime
import matplotlib.pyplot as plt

# Función para limpiar datos
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

# Función para entrenar un modelo LSTM
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

# Función para optimizar el modelo SVM
def optimize_svm(X_train, y_train):
    param_grid = {'C': [0.1, 1, 10], 'gamma': [1, 0.1, 0.01]}
    grid = GridSearchCV(SVR(), param_grid, refit=True, cv=5)
    grid.fit(X_train, y_train)
    return grid.best_estimator_

# Función para plotear las predicciones
def plot_forecast(dates_test, y_test, svm_predictions, lstm_predictions, combined_predictions):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(dates_test, y_test, label='Precio Real')
    ax.plot(dates_test, svm_predictions, label='Predicciones SVM')
    ax.plot(dates_test, lstm_predictions, label='Predicciones LSTM')
    ax.plot(dates_test, combined_predictions, label='Predicciones Combinadas')
    ax.set_title('Predicción de Precios de Acciones')
    ax.set_xlabel('Fecha')
    ax.set_ylabel('Precio Normalizado')
    ax.legend()
    st.pyplot(fig)

    st.write("""
    En el gráfico anterior, se muestran tres series de datos:

    - **Precio Real:** Estos son los precios de cierre reales de las acciones para el período de prueba.
    - **Predicciones SVM:** Estas son las predicciones del modelo de Máquinas de Vectores de Soporte (SVM). Este modelo es muy efectivo para encontrar patrones complejos en los datos.
    - **Predicciones LSTM:** Estas son las predicciones del modelo de Long Short-Term Memory (LSTM). Las LSTM son un tipo de red neuronal recurrente que se especializa en datos secuenciales, como las series temporales.
    - **Predicciones Combinadas:** Estas predicciones son la mediana de las predicciones SVM y LSTM, lo que ayuda a reducir el error de predicción combinando la fortaleza de ambos modelos.
    """)


# Función para generar predicciones y métricas
def generate_predictions(ticker, start_date, end_date):
    """
    Genera predicciones utilizando modelos SVM y LSTM entrenados y devuelve métricas de evaluación.
    Además, predice la tendencia (subida o bajada) del precio para mañana.
    """
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
    target_column = 'Close'  # Columna objetivo para la predicción
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

    # Reshape los datos para el modelo LSTM
    X_train_lstm = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test_lstm = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))

    # Entrenar los modelos
    svm_model = optimize_svm(X_train, y_train)
    lstm_model = train_lstm(X_train_lstm, y_train, (1, X_train.shape[1]))

    # Generar predicciones
    svm_predictions = pd.Series(svm_model.predict(X_test), index=X_test.index)
    lstm_predictions = pd.Series(lstm_model.predict(X_test_lstm).flatten(), index=X_test.index)
    combined_predictions = pd.Series(np.median([svm_predictions, lstm_predictions], axis=0), index=X_test.index)

    # Calcular las predicciones para mañana
    last_close_price = data.iloc[-1]['Close']  # Precio de cierre más reciente
    tomorrow_features = X_test.iloc[-1].values.reshape(1, -1)  # Características para mañana

    # Predicción LSTM para mañana
    tomorrow_features_reshaped = tomorrow_features.reshape(1, 1, tomorrow_features.shape[1])
    lstm_prediction_tomorrow = lstm_model.predict(tomorrow_features_reshaped)[0][0]

    # Determinar si se espera que el precio suba o baje mañana y por cuánto
    price_change = lstm_prediction_tomorrow - last_close_price
    if lstm_prediction_tomorrow > last_close_price:
        tomorrow_trend = f"subida de {price_change:.2f}"
    elif lstm_prediction_tomorrow < last_close_price:
        tomorrow_trend = f"bajada de {price_change:.2f}"
    else:
        tomorrow_trend = "sin cambio"

    # Calcular métricas de evaluación
    mape_svm = mean_absolute_percentage_error(y_test, svm_predictions)
    mape_lstm = mean_absolute_percentage_error(y_test, lstm_predictions)
    mape_combined = mean_absolute_percentage_error(y_test, combined_predictions)
    rmse_svm = np.sqrt(mean_squared_error(y_test, svm_predictions))
    rmse_lstm = np.sqrt(mean_squared_error(y_test, lstm_predictions))
    rmse_combined = np.sqrt(mean_squared_error(y_test, combined_predictions))

    return (svm_predictions, lstm_predictions, combined_predictions,
            mape_svm, mape_lstm, mape_combined,
            rmse_svm, rmse_lstm, rmse_combined,
            dates_test, y_test, tomorrow_trend)


# Función para mostrar la página de ensamblado
def mostrar_pagina_ensamblado():
    st.title("Modelo ensamblado")
    st.write("""
    Este sistema se basa en el análisis de datos históricos de precios de acciones para predecir su comportamiento futuro. Se utilizan dos modelos de aprendizaje automático:
        
    """)
    st.write(
        """
        - **SVM (Support Vector Machine):** Un algoritmo robusto que busca patrones en los datos para clasificar o predecir valores futuros.
        - **LSTM (Long Short-Term Memory):** Un tipo de red neuronal recurrente que se destaca en el manejo de datos secuenciales, como series temporales de precios.
        """
    )
    
    # Entrada de usuario para el ticker del instrumento financiero
    ticker = st.text_input("Ticker del instrumento financiero", value='FSM')
    # Entrada de usuario para las fechas de inicio y fin
    start_date = st.date_input("Fecha de inicio", value=datetime.date(2022, 1, 1))
    end_date = st.date_input("Fecha de fin", value=datetime.date(2023, 8, 11))

    # Botón para ejecutar el análisis
    if st.button("Ejecutar Análisis"):
        (svm_predictions, lstm_predictions, combined_predictions, mape_svm, 
        mape_lstm, mape_combined, rmse_svm, rmse_lstm, rmse_combined, 
        dates_test, y_test, tomorrow_trend) = generate_predictions(ticker, start_date, end_date)

        # Mostrar métricas de evaluación
        st.write("### Métricas de evaluación")
        st.write(f'MAPE (Mean Absolute Percentage Error) del modelo SVM: {mape_svm:.2f}')
        st.write(f'MAPE (Mean Absolute Percentage Error) del modelo LSTM: {mape_lstm:.2f}')
        st.write(f'MAPE (Mean Absolute Percentage Error) del modelo combinado: {mape_combined:.2f}')
        st.write(f'RMSE (Root Mean Squared Error) del modelo SVM: {rmse_svm:.2f}')
        st.write(f'RMSE (Root Mean Squared Error) del modelo LSTM: {rmse_lstm:.2f}')
        st.write(f'RMSE (Root Mean Squared Error) del modelo combinado: {rmse_combined:.2f}')

        # Mostrar predicciones numéricas en un DataFrame
        st.write("### Predicciones numéricas")
        predictions_df = pd.DataFrame({'Actual': y_test, 'SVM Predicted': svm_predictions, 
                                       'LSTM Predicted': lstm_predictions, 'Combined Predicted': combined_predictions})
        st.dataframe(predictions_df)

        # Graficar las predicciones junto con los valores reales
        st.write("### Gráfico de Predicciones")
        plot_forecast(dates_test, y_test, svm_predictions, lstm_predictions, combined_predictions)

        # Mostrar la predicción de tendencia para mañana
        st.write("### Predicción de tendencia para mañana")
        st.write(f"Se espera que el precio de mañana tenga una tendencia de {tomorrow_trend}.")
        if "subida" in tomorrow_trend or "bajada" in tomorrow_trend:
            st.write(f"Se espera que el precio cambie aproximadamente {tomorrow_trend.split('de')[1].strip()} unidades.")

# Función principal para la aplicación Streamlit
def main():
    pages = {
        "Modelo ensamblado": mostrar_pagina_ensamblado,
    }
    st.sidebar.title("Navegación")
    selection = st.sidebar.radio("Seleccione una página:", list(pages.keys()))
    pages[selection]()

if __name__ == "__main__":
    main()

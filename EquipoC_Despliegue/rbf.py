import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from scipy.linalg import pinv
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Funciones adicionales para el modelo de redes neuronales
def load_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

def prepare_data(data):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[['Close']])
    return scaled_data, scaler

def create_model():
    model = Sequential()
    model.add(Dense(50, activation='relu', input_dim=1))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def train_model(model, X_train, y_train, epochs=50, batch_size=32):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    return model

def make_predictions(model, X):
    predictions = model.predict(X)
    return predictions

def inverse_transform(predictions, scaler):
    return scaler.inverse_transform(predictions)

# Función principal de la aplicación
def mostrar_pagina_rbf():
    # Título de la aplicación
    st.title("Modelo RBF - Análisis Financiero con Streamlit")

    st.markdown("""
    Esta aplicación permite realizar un análisis financiero utilizando modelos de machine learning. 
    Podrás seleccionar el modelo a ejecutar, la acción o instrumento financiero que deseas analizar, y el rango de fechas. 
    Se mostrarán gráficos de los precios reales, la media móvil, y las predicciones realizadas por el modelo. 
    También se proporcionarán recomendaciones basadas en los resultados obtenidos.
    """)

    # Selección de la acción o instrumento financiero
    ticker = st.text_input("Ingrese el ticker de la acción o instrumento financiero", "FSM")
    st.write("""
    **Ticker:** Un ticker es un símbolo único utilizado para identificar una acción o instrumento financiero en el mercado.
    Por ejemplo, 'FSM' corresponde a Fortuna Silver Mines Inc. Introduce el ticker del instrumento que deseas analizar.
    """)

    # Selección del rango de fechas
    start_date = st.date_input("Fecha de inicio", pd.to_datetime("2019-01-01"))
    end_date = st.date_input("Fecha de fin", pd.to_datetime("2023-12-31"))
    st.write("""
    **Rango de fechas:** Selecciona el rango de fechas para el cual deseas realizar el análisis.
    El análisis incluirá todos los datos disponibles desde la fecha de inicio hasta la fecha de fin.
    """)

    # Descargar datos
    st.write("### Descargando datos...")
    data = yf.download(ticker, start=start_date, end=end_date)
    st.write("### Datos descargados:")
    st.write(data.head())
    st.write("""
    Los datos descargados muestran información financiera sobre el instrumento seleccionado.
    Incluyen el precio de apertura (Open), el precio más alto del día (High), el precio más bajo del día (Low), el precio de cierre (Close), el volumen de transacciones (Volume), y el precio ajustado de cierre (Adj Close).
    """)

    # Gráfico de los precios reales
    st.write("### Gráfico de los precios reales")
    fig, ax = plt.subplots()
    ax.plot(data['Close'], label='Precio de cierre')
    ax.set_xlabel('Fecha')
    ax.set_ylabel('Precio')
    ax.set_title('Precio de cierre del instrumento financiero')
    ax.legend()
    st.pyplot(fig)
    st.write("""
    **Gráfico de precios reales:** Este gráfico muestra la evolución del precio de cierre del instrumento financiero a lo largo del tiempo.
    El precio de cierre es el precio al cual se realizó la última transacción del día.
    """)

    # Calcular la media móvil y graficarla
    st.write("### Gráfico de la media móvil")
    data['Media Movil'] = data['Close'].rolling(window=20).mean()
    fig, ax = plt.subplots()
    ax.plot(data['Close'], label='Precio de cierre')
    ax.plot(data['Media Movil'], label='Media Móvil (20 días)')
    ax.set_xlabel('Fecha')
    ax.set_ylabel('Precio')
    ax.set_title('Precio de cierre y Media Móvil')
    ax.legend()
    st.pyplot(fig)
    st.write("""
    **Gráfico de media móvil:** Este gráfico incluye la media móvil de 20 días del precio de cierre.
    La media móvil es una técnica de análisis utilizada para suavizar las fluctuaciones en los datos y mostrar la tendencia subyacente.
    """)

    # Selección del modelo a ejecutar
    model_option = st.selectbox("Seleccione el modelo a ejecutar", ("Modelo Lineal", "Redes Neuronales", "SVM"))
    st.write("""
    **Modelos a ejecutar:**
    - **Modelo Lineal:** Un modelo simple que asume una relación lineal entre las variables.
    - **Redes Neuronales:** Modelos más complejos que pueden capturar relaciones no lineales en los datos.
    - **SVM:** Máquinas de soporte vectorial, otra técnica avanzada para la clasificación y regresión.
    Selecciona el modelo que deseas utilizar para realizar las predicciones.
    """)

    # Preprocesamiento de datos
    st.write("### Preprocesamiento de datos")
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data[['Close']])
    st.write("""
    **Preprocesamiento de datos:** Los datos se normalizan utilizando MinMaxScaler para escalar los precios de cierre en un rango entre 0 y 1.
    Esto ayuda a mejorar el rendimiento de los modelos de machine learning.
    """)

    # División de datos en entrenamiento y prueba
    train_size = int(len(data_scaled) * 0.8)
    train_data, test_data = data_scaled[:train_size], data_scaled[train_size:]
    st.write(f"Datos de entrenamiento: {train_size} datos")
    st.write(f"Datos de prueba: {len(data_scaled) - train_size} datos")
    st.write("""
    **División de datos:** Los datos se dividen en un conjunto de entrenamiento (80%) y un conjunto de prueba (20%).
    El conjunto de entrenamiento se utiliza para entrenar el modelo, y el conjunto de prueba se utiliza para evaluar su rendimiento.
    """)

    # Función para entrenar y predecir usando un modelo lineal simple
    def train_predict_linear(train, test):
        X_train, y_train = train[:-1], train[1:]
        X_test, y_test = test[:-1], test[1:]
        model = np.linalg.pinv(X_train) @ y_train
        predictions = X_test @ model
        return predictions, y_test

    # Función para entrenar y predecir usando SVM
    def train_predict_svm(train, test):
        X_train, y_train = train[:-1], train[1:]
        X_test, y_test = test[:-1], test[1:]
        model = SVR(kernel='rbf')
        model.fit(X_train, y_train.ravel())
        predictions = model.predict(X_test)
        return predictions, y_test

    # Función para mostrar los resultados y las predicciones
    def display_results(predictions, actual):
        st.write("### Resultados de las predicciones")
        st.write("Valores predichos:")
        st.write(predictions)
        st.write("Valores reales:")
        st.write(actual)

        fig, ax = plt.subplots()
        ax.plot(predictions, label='Predicciones')
        ax.plot(actual, label='Valores Reales')
        ax.set_xlabel('Tiempo')
        ax.set_ylabel('Precio')
        ax.set_title('Comparación entre Predicciones y Valores Reales')
        ax.legend()
        st.pyplot(fig)

        mse = mean_squared_error(actual, predictions)
        st.write(f"Error cuadrático medio de las predicciones: {mse}")

        st.write("### Recomendación")
        st.write("""
        Basado en el análisis, se recomienda observar la tendencia de las predicciones y los valores reales para tomar decisiones informadas sobre la compra o venta del instrumento financiero.
        Si las predicciones muestran una tendencia al alza, podría ser un buen momento para comprar. Si muestran una tendencia a la baja, podría ser mejor esperar o vender.
        """)

    # Ejecutar el modelo seleccionado
    if model_option == "Modelo Lineal":
        st.write("### Ejecutando modelo lineal...")
        predictions, actual = train_predict_linear(train_data, test_data)
        display_results(predictions, actual)
    elif model_option == "Redes Neuronales":
        st.write("### Ejecutando modelo de Redes Neuronales...")
        scaled_data, scaler = prepare_data(data)
        X = scaled_data[:-1]
        y = scaled_data[1:]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        model = create_model()
        model = train_model(model, X_train, y_train)
        predictions = make_predictions(model, X_test)
        predictions_actual = inverse_transform(predictions, scaler)
        display_results(predictions_actual, scaler.inverse_transform(y_test))
    elif model_option == "SVM":
        st.write("### Ejecutando modelo SVM...")
        predictions, actual = train_predict_svm(train_data, test_data)
        display_results(predictions, actual)

    # Predicción del precio de mañana
    st.subheader('Predicción del Precio de Mañana')
    last_scaled_price = data_scaled[-1].reshape(1, -1)
    if model_option == "Modelo Lineal":
        tomorrow_prediction = np.dot(last_scaled_price, np.linalg.pinv(train_data[:-1]) @ train_data[1:])
    elif model_option == "SVM":
        model = SVR(kernel='rbf')
        model.fit(train_data[:-1], train_data[1:].ravel())
        last_scaled_price = data_scaled[-1].reshape(1, -1)
        tomorrow_prediction = model.predict(last_scaled_price)
    elif model_option == "Redes Neuronales":
        last_scaled_price = data_scaled[-1].reshape(-1, 1)
        tomorrow_prediction = model.predict(last_scaled_price)
    else:
        tomorrow_prediction = None  # Placeholder para cuando se implementen otros modelos

    # Asegurarse de que tomorrow_prediction sea un array numpy
    tomorrow_prediction = np.array(tomorrow_prediction).reshape(-1, 1)
    tomorrow_price = scaler.inverse_transform(tomorrow_prediction)[0][0]

    st.write(f'La predicción del precio para mañana es: **{tomorrow_price:.2f}**')
    st.write(f'El precio de cierre actual es: **{data["Close"].iloc[-1]:.2f}**')

if __name__ == '__main__':
    mostrar_pagina_rbf()

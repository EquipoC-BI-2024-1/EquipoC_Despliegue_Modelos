import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import yfinance as yf
from datetime import datetime, timedelta

@st.cache
def obtener_datos_financieros(ticker, start, end):
    datos_originales = yf.download(ticker, start=start, end=end)
    datos = datos_originales.copy()  # Hacemos una copia de los datos originales
    datos.reset_index(inplace=True)
    return datos, datos_originales  # Devolvemos tanto los datos copiados como los originales

def entrenar_y_predecir_rfr(datos):
    datos['Fecha'] = pd.to_datetime(datos['Date'])
    datos.set_index('Fecha', inplace=True)
    datos['Año'] = datos.index.year
    datos['Mes'] = datos.index.month
    datos['Día'] = datos.index.day

    X = datos[['Año', 'Mes', 'Día']]
    y = datos['Close']

    modelo = RandomForestRegressor(n_estimators=100, random_state=42)
    modelo.fit(X, y)

    predicciones = modelo.predict(X)
    pred_df = datos[['Close']].copy()
    pred_df['Predicted'] = predicciones

    return pred_df, modelo

def entrenar_y_predecir_rfc(datos):
    datos['Fecha'] = pd.to_datetime(datos['Date'])
    datos.set_index('Fecha', inplace=True)
    datos['Año'] = datos.index.year
    datos['Mes'] = datos.index.month
    datos['Día'] = datos.index.day
    datos['Target'] = np.where(datos['Close'].shift(-1) > datos['Close'], 1, 0)

    X = datos[['Año', 'Mes', 'Día']]
    y = datos['Target']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X_scaled, y)

    predicciones = modelo.predict(X_scaled)
    pred_df = datos[['Close']].copy()
    pred_df['Predicted'] = predicciones

    return pred_df, modelo, scaler

def calcular_media_movil(datos, ventana):
    return datos.rolling(window=ventana).mean()

def mostrar_pagina_rfr():
    st.title('Modelo Random Forest')

    st.markdown("""
    Esta aplicación permite analizar y predecir los precios de instrumentos financieros. 
    Puedes seleccionar la acción o instrumento que deseas analizar, así como el rango de fechas.
    """)

    st.sidebar.header('Parámetros de entrada')
    ticker = st.sidebar.text_input('Ticker del instrumento financiero', 'AAPL')
    start_date = st.sidebar.date_input('Fecha de inicio', datetime(2020, 1, 1))
    end_date = st.sidebar.date_input('Fecha de fin', datetime(2023, 1, 1))

    datos, datos_originales = obtener_datos_financieros(ticker, start_date, end_date)
    datos = datos.copy()  # Clonamos el DataFrame para evitar mutaciones del cache

    st.subheader('Datos Financieros')
    st.write(datos)

    st.subheader('Ploteo de los Precios Reales')
    fig_real = px.line(datos, x='Date', y='Close', title='Precios Reales')
    st.plotly_chart(fig_real)

    st.subheader('Media Móvil de los Precios Reales')
    window_size = st.sidebar.slider('Tamaño de la ventana para la media móvil', 5, 100, 20)
    datos_modificados = datos.copy()  # Hacemos una copia de los datos para modificar
    media_movil = calcular_media_movil(datos_modificados['Close'], window_size)
    fig_media_movil = px.line(x=media_movil.index, y=media_movil, title='Media Móvil')
    st.plotly_chart(fig_media_movil)

    # Predicción del precio de mañana con modelo de regresión
    predicciones_rfr, modelo_rfr = entrenar_y_predecir_rfr(datos)

    st.subheader('Predicciones de Precios (Regresión)')
    st.write(predicciones_rfr)

    st.subheader('Ploteo de las Predicciones de Precios')
    fig_pred_rfr = px.line(predicciones_rfr, x=predicciones_rfr.index, y='Predicted', title='Predicciones de Precios')
    st.plotly_chart(fig_pred_rfr)

    st.subheader('Comparativa entre Precios Reales y Predicciones de Precios')
    fig_comparativa_rfr = px.line(title='Comparativa de Precios Reales y Predicciones de Precios')
    fig_comparativa_rfr.add_scatter(x=predicciones_rfr.index, y=predicciones_rfr['Close'], mode='lines', name='Precios Reales')
    fig_comparativa_rfr.add_scatter(x=predicciones_rfr.index, y=predicciones_rfr['Predicted'], mode='lines', name='Predicciones')
    st.plotly_chart(fig_comparativa_rfr)

    ultimo_dia = datos_originales.index[-1] + timedelta(days=1)
    caracteristicas_siguiente_dia = [[ultimo_dia.year, ultimo_dia.month, ultimo_dia.day]]
    prediccion_precio_siguiente_dia = modelo_rfr.predict(caracteristicas_siguiente_dia)[0]

    st.subheader('Predicción del Precio de Mañana (Regresión)')
    st.markdown(f"""
    El precio de {ticker} se pronostica según el modelo Random Forest Regressor para el siguiente día como: **${prediccion_precio_siguiente_dia:.2f}** por acción.

    Basado en esta predicción, se recomienda:
    """)

    if prediccion_precio_siguiente_dia > datos_originales['Close'].iloc[-1]:
        st.markdown("- **Considerar la compra** debido a la tendencia al alza pronosticada.")
    else:
        st.markdown("- **Considerar la venta** o esperar debido a la tendencia a la baja pronosticada.")

    # Predicción de la tendencia de mañana con modelo de clasificación
    predicciones_rfc, modelo_rfc, scaler = entrenar_y_predecir_rfc(datos)

    st.subheader('Predicciones de Tendencia (Clasificación)')
    predicciones_tendencia = predicciones_rfc.copy()
    predicciones_tendencia['Tendencia'] = np.where(predicciones_tendencia['Predicted'] == 1, 'Subida', 'Bajada')
    st.write(predicciones_tendencia)

    st.subheader('Ploteo de las Predicciones de Tendencia')
    fig_pred_rfc = px.line(predicciones_tendencia, x=predicciones_tendencia.index, y='Tendencia', title='Predicciones de Tendencia')
    st.plotly_chart(fig_pred_rfc)

    st.subheader('Comparativa entre Precios Reales y Predicciones de Tendencia')
    fig_comparativa_rfc = px.line(title='Comparativa de Precios Reales y Predicciones de Tendencia')
    fig_comparativa_rfc.add_scatter(x=predicciones_rfc.index, y=predicciones_rfc['Close'], mode='lines', name='Precios Reales')
    fig_comparativa_rfc.add_scatter(x=predicciones_tendencia.index, y=predicciones_tendencia['Tendencia'], mode='lines', name='Predicciones de Tendencia')
    st.plotly_chart(fig_comparativa_rfc)

    caracteristicas_siguiente_dia_scaled = scaler.transform(caracteristicas_siguiente_dia)
    prediccion_tendencia_siguiente_dia = modelo_rfc.predict(caracteristicas_siguiente_dia_scaled)[0]
    tendencia_siguiente_dia = 'Subida' if prediccion_tendencia_siguiente_dia == 1 else 'Bajada'

    st.subheader('Predicción de la Tendencia de Mañana (Clasificación)')
    st.markdown(f"""
    La tendencia del precio de {ticker} se pronostica según el modelo Random Forest Classifier para el siguiente día como: **{tendencia_siguiente_dia}**.

    Basado en esta predicción, se recomienda:
    """)

    if tendencia_siguiente_dia == 'Subida':
        st.markdown("- **Considerar la compra** debido a la tendencia al alza pronosticada.")
    else:
        st.markdown("- **Considerar la venta** o esperar debido a la tendencia a la baja pronosticada.")

    st.markdown("""
    ### Información Adicional
    - Los datos mostrados incluyen los precios históricos de cierre del instrumento seleccionado.
    - La media móvil se utiliza para suavizar las fluctuaciones en los datos y resaltar la tendencia.
    - Las predicciones de precio se basan en un modelo Random Forest Regressor que utiliza características derivadas de las fechas para proyectar los futuros precios.
    - Las predicciones de tendencia se basan en un modelo Random Forest Classifier que clasifica la dirección del precio del siguiente día.
    - Es importante considerar otros factores externos y no depender únicamente de las predicciones para tomar decisiones de inversión.
    """)

    st.sidebar.markdown("""
    ## Acerca de esta aplicación
    Esta aplicación fue desarrollada para proporcionar una herramienta interactiva para el análisis y predicción de precios de instrumentos financieros. 
    Las predicciones deben ser utilizadas con precaución y no garantizan resultados futuros. Se recomienda consultar con un asesor financiero antes de tomar decisiones de inversión.
    """)

if __name__ == "__main__":
    mostrar_pagina_rfr()

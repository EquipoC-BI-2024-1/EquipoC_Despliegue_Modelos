import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import datetime

# Función para cargar y procesar los datos
def load_data(ticker, start_date, end_date):
    url = f'https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={start_date}&period2={end_date}&interval=1d&events=history&includeAdjustedClose=true'
    data = pd.read_csv(url)
    data = data.drop_duplicates()
    
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    non_numeric_cols = data.select_dtypes(exclude=[np.number]).columns
    
    imputer = SimpleImputer(strategy='mean')
    data_imputed_numeric = pd.DataFrame(imputer.fit_transform(data[numeric_cols]), columns=numeric_cols)
    
    data_imputed = pd.concat([data_imputed_numeric, data[non_numeric_cols].reset_index(drop=True)], axis=1)
    
    scaler = StandardScaler()
    data_scaled_numeric = pd.DataFrame(scaler.fit_transform(data_imputed_numeric), columns=numeric_cols)
    
    data_scaled = pd.concat([data_scaled_numeric, data[non_numeric_cols].reset_index(drop=True)], axis=1)
    
    return data, data_scaled

# Función para entrenar el modelo SVR
def train_model_svr(data_scaled, target_column_name):
    target_corr = data_scaled.corr()[target_column_name].abs().sort_values(ascending=False)
    relevant_features = target_corr[target_corr > 0.1].index.tolist()
    data_relevant = data_scaled[relevant_features]
    
    X = data_relevant.drop(columns=[target_column_name])
    y = data_relevant[target_column_name]
    
    model = SVR()
    model.fit(X, y)
    
    return model, data_relevant

# Función para predecir y mostrar resultados para SVR
def show_regression_predictions(model, data_relevant, target_column_name):
    X = data_relevant.drop(columns=[target_column_name])
    y_true = data_relevant[target_column_name]
    y_pred = model.predict(X)
    
    st.write("### Predicciones numéricas (Modelo SVR)")
    st.write("""
    En esta tabla, se muestran las predicciones generadas por el modelo SVR junto con los valores reales. 
    Esto permite comparar directamente cuánto difieren las predicciones de los valores reales.
    """)
    predictions_df = pd.DataFrame({'Actual': y_true, 'Predicted': y_pred})
    st.dataframe(predictions_df)
    
    st.write("### Gráfico de Predicciones (Modelo SVR)")
    st.write("""
    El siguiente gráfico muestra los valores reales y las predicciones realizadas por el modelo SVR. 
    Este gráfico ayuda a visualizar el desempeño del modelo y ver cómo de cerca las predicciones siguen la tendencia de los datos reales.
    """)
    plt.figure(figsize=(10, 5))
    plt.plot(y_true.values, label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.legend()
    st.pyplot(plt)
    
    st.write("### Recomendación (Modelo SVR)")
    st.write("""
    **Recomendación:** Basado en los resultados obtenidos con el modelo SVR, se sugiere realizar un análisis más profundo considerando otros modelos o ajustando los parámetros para mejorar la precisión de las predicciones. 
    También es crucial complementar estos análisis con un contexto financiero más amplio y actualizado.
    """)

def mostrar_pagina_svm():
    st.title("Análisis Predictivo de Instrumentos Financieros con SVR")
    
    st.write("""
    En esta aplicación, se utiliza Support Vector Regression (SVR) para realizar análisis predictivo sobre los precios de cierre de instrumentos financieros.
    """)
    
    ticker = st.text_input("Ticker del Instrumento Financiero", value='AAPL')
    start_date = st.date_input("Fecha de Inicio", value=datetime.datetime(2021, 1, 1))
    end_date = st.date_input("Fecha de Fin", value=datetime.datetime(2021, 8, 11))
    
    start_date_timestamp = int(datetime.datetime.combine(start_date, datetime.datetime.min.time()).timestamp())
    end_date_timestamp = int(datetime.datetime.combine(end_date, datetime.datetime.min.time()).timestamp())
    
    data, data_scaled = load_data(ticker, start_date_timestamp, end_date_timestamp)
    
    if st.button("Ejecutar Análisis"):
        st.write("### Análisis Exploratorio de Datos")
        
        st.write("#### Precios de Cierre Históricos")
        st.write("""
        El siguiente gráfico muestra los precios de cierre históricos del instrumento financiero seleccionado. 
        Esto ayuda a entender la tendencia general y la volatilidad del precio durante el período seleccionado.
        """)
        st.line_chart(data['Close'])
        
        st.write("#### Media Móvil de Precios de Cierre")
        st.write("""
        La media móvil de 20 días se utiliza para suavizar las fluctuaciones a corto plazo y resaltar las tendencias a largo plazo en los precios de cierre.
        """)
        data['MA'] = data['Close'].rolling(window=20).mean()
        st.line_chart(data[['Close', 'MA']])
        
        st.write("### Predicciones y Resultados")
        
        st.write("#### Modelo SVR - Predicción de Precio para Mañana")
        model_svr, data_relevant_svr = train_model_svr(data_scaled, 'Close')
        show_regression_predictions(model_svr, data_relevant_svr, 'Close')

if __name__ == "__main__":
    mostrar_pagina_svm()

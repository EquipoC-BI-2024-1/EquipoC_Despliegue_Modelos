# **EqC_2024_1_Despliegue_en_la_Web_de_los_Modelos_de_Python_de_Predicción_de_Acciones_Financieras_con_Streamlit**

# **Integrantes**

* Callupe Arias Jefferson Jesus
* Durand Caracuzma Marlon Milko
* Huarhua Piñas Edson Sebastian
* Ovalle Martinez Lisett Andrea
* Romero Cisneros Karlo

## **Descripción del Proyecto**
Este proyecto tiene como objetivo desplegar en la web varios modelos de predicción de acciones financieras utilizando Streamlit. Los modelos desarrollados en Python son capaces de predecir el comportamiento futuro de las acciones financieras basándose en datos históricos.

## **Modelos Desplegados**
Los modelos que se desplegarán son:

* **LSTM regresor:** Un modelo de redes neuronales recurrentes de memoria a largo plazo, utilizado para la predicción de series temporales.
* **SVR regresor:** Soporte Vectorial de Regresión, una técnica basada en máquinas de soporte vectorial adaptada para problemas de regresión.
* **RBF regresor:** Función de Base Radial, un método que utiliza funciones de base radial para la regresión.
* **RandomForest regresor:** Un modelo de conjunto que utiliza múltiples árboles de decisión para mejorar la precisión de la predicción.
* **Ensamblado LSTM SVR regresor:** Un modelo de conjunto que combina las predicciones de LSTM y SVR para mejorar el rendimiento.
* **SVC clasificador:** Soporte Vectorial de Clasificación, utilizado para clasificar las acciones en diferentes categorías basándose en características específicas.

## **Requisitos del Sistema**

Para ejecutar este proyecto, necesitarás tener instaladas las siguientes dependencias:

Python 3.10
Streamlit
TensorFlow
scikit-learn
numpy
pandas
matplotlib
otros

Puedes instalar las dependencias necesarias ejecutando el siguiente comando:

```pip install -r requirements.txt```

## **Ejecución del Proyecto**

Para ejecutar la aplicación, sigue los siguientes pasos:

1. Clona este repositorio:
   
```git clone https://github.com/EquipoC-BI-2024-1/EqC_DespliegueModelosPython.git```

2. Navega al directorio del proyecto:
   
```cd EqC_Despliegue/Pages```

3. Ejecuta la aplicación de Streamlit:

```streamlit run inicio.py```

## **Uso de la Aplicación**
Una vez ejecutada la aplicación, se abrirá una interfaz web en tu navegador donde podrás interactuar con los diferentes modelos de predicción. Podrás cargar tus propios datos de acciones financieras y visualizar las predicciones generadas por cada modelo.

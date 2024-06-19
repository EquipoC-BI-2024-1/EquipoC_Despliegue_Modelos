import streamlit as st

# La primera llamada a Streamlit debe ser st.set_page_config()
st.set_page_config(
    page_title="Despliegue web",
    page_icon="👨🏻‍💻",
)

# Importa las funciones desde los archivos en Pages
from ensamblado import mostrar_pagina_ensamblado
from lstm import mostrar_pagina_lstm
from svr import mostrar_pagina_svm  # Asegúrate de importar la función correcta de svr.py
from svc import mostrar_pagina_svc
from rfr_de_regresion import mostrar_pagina_rfr  # Importa directamente desde rfr_de_regresion.py

def mostrar_pagina_inicio():
    st.sidebar.title('Menú')

    # Widget interactivo para seleccionar la página
    page = st.sidebar.radio('Selecciona una opción del menú', 
                            ["PRESENTACIÓN", "MODELO ENSAMBLADO", "MODELO LSTM", "MODELO SVR", "MODELO RBF", "MODELO RFR", "MODELO SVC"])

    # Contenido según la página seleccionada
    if page == "PRESENTACIÓN":
        with st.container():
            st.title("UNIVERSIDAD NACIONAL MAYOR DE SAN MARCOS")
            st.header("Facultad de Ingeniería de Sistemas e Informática")
        
        st.write("\n\n\n\n\n\n\n\n")
        # Centrar la imagen usando columnas
        col1, col2, col3 = st.columns(3)
        with col2:
            st.image("https://seeklogo.com/images/U/universidad-nacional-mayor-de-san-marcos-logo-5BECFDDBD8-seeklogo.com.png", width=150)

        st.header("\n\n\n\n\n¡Despliegue de Modelos del Equipo C!")
        st.write("---")
        st.subheader("Equipo C está compuesto por:")
        
        equipo = """
        - **Callupe Arias** - Jefferon Jesus
        - **Durand Caracuzma** - Marlon Milko
        - **Huarhua Piñas** - Edson Sebastian
        - **Ovalle Martinez** - Lisett Andrea
        - **Romero Cisneros** - Karlo Brandi
        """
        st.markdown(equipo)
        
    elif page == "MODELO ENSAMBLADO":
        mostrar_pagina_ensamblado()

    elif page == "MODELO LSTM":
        mostrar_pagina_lstm()
        
    elif page == "MODELO SVR":
         mostrar_pagina_svm()
    
    elif page == "MODELO RBF":
         from rbf import mostrar_pagina_rbf  # Importa la función de la página RBF
         mostrar_pagina_rbf()

    elif page == "MODELO SVC":
         mostrar_pagina_svc()
    
    elif page == "MODELO RFR":
         mostrar_pagina_rfr()
        
if __name__ == "__main__":
    mostrar_pagina_inicio()

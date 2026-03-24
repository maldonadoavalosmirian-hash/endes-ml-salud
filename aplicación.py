import streamlit as st  # <--- ESTA LÍNEA ES LA QUE FALTA
import pandas as pd
import joblib

# 1. Configuración de la página
st.set_page_config(page_title="Análisis Predictivo de Salud Infantil", layout="wide")

# 2. Cargar los modelos
try:
    modelo_reg = joblib.load('mejor_modelo_reg.pkl')
    modelo_clf = joblib.load('mejor_modelo_clf.pkl')
except Exception as e:
    st.error(f"Error al cargar modelos: {e}")

# 3. Interfaz de Usuario
st.title("🩺 Análisis Predictivo de Salud Infantil (ENDES 2024)")

# --- BARRA LATERAL CON RANGOS DEL DICCIONARIO ---
st.sidebar.header("📥 Datos de Entrada")
proyecto = st.sidebar.radio("Seleccione el Tipo de Análisis", 
                           ["Regresión (Nivel de Hemoglobina)", 
                            "Clasificación (Diagnóstico de Anemia)"])

st.sidebar.markdown("---")
# HW1: Edad en meses (Rango 0:59)
v_edad = st.sidebar.slider("Edad del niño (meses) [HW1]", 0, 59, 24)

# HW2: Peso (Rango 15:500 -> 1.5kg a 50kg)
v_peso = st.sidebar.number_input("Peso en kg (Ej: 9.5) [HW2]", 
                                min_value=1.5, 
                                max_value=50.0, 
                                value=12.0, 
                                step=0.1)

# HW3: Talla (Rango 400:1500 -> 40cm a 150cm)
v_talla = st.sidebar.number_input("Talla en cm (Ej: 70.5) [HW3]", 
                                 min_value=40.0, 
                                 max_value=150.0, 
                                 value=85.0, 
                                 step=0.1)

# Preparación de datos (Multiplicamos por 10 para HW2 y HW3 según formato REC44)
datos_input = pd.DataFrame([[v_edad, v_peso * 10, v_talla * 10]], 
                            columns=['HW1', 'HW2', 'HW3'])

# 4. Botón y Lógica de Predicción
if st.sidebar.button("PREDECIR", type="primary"):
    if proyecto == "Regresión (Nivel de Hemoglobina)":
        pred_hemo = modelo_reg.predict(datos_input)[0]
        st.subheader("📊 Resultado de Regresión")
        st.metric("Hemoglobina Estimada", f"{pred_hemo/10:.1f} g/dL")
    else:
        pred_anemia = modelo_clf.predict(datos_input)[0]
        st.subheader("🎯 Resultado de Clasificación")
        if pred_anemia == 1:
            st.error("⚠️ DIAGNÓSTICO: PROBABLE ANEMIA")
        else:
            st.success("✅ DIAGNÓSTICO: SIN ANEMIA")

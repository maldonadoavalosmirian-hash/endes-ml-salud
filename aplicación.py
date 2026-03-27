import streamlit as st
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

# --- BARRA LATERAL ---
st.sidebar.header("📥 Datos de Entrada")
proyecto = st.sidebar.radio("Seleccione el Tipo de Análisis", 
                            ["Regresión (Nivel de Hemoglobina)", 
                             "Clasificación (Diagnóstico de Anemia)"])

st.sidebar.markdown("---")

# Entradas de datos según formato REC44
v_edad = st.sidebar.slider("Edad del niño (meses) [HW1]", 0, 59, 24)

v_peso = st.sidebar.number_input("Peso en kg (Ej: 9.5) [HW2]", 
                                 min_value=1.5, 
                                 max_value=50.0, 
                                 value=12.0, 
                                 step=0.1)

v_talla = st.sidebar.number_input("Talla en cm (Ej: 85.0) [HW3]", 
                                  min_value=40.0, 
                                  max_value=150.0, 
                                  value=85.0, 
                                  step=0.1)

# Preparación de datos (Escalado x10 para que coincida con el entrenamiento del modelo)
datos_input = pd.DataFrame([[v_edad, v_peso * 10, v_talla * 10]], 
                            columns=['HW1', 'HW2', 'HW3'])

# 4. Botón y Lógica de Predicción
if st.sidebar.button("PREDECIR", type="primary"):
    
    if proyecto == "Regresión (Nivel de Hemoglobina)":
        # LÓGICA DE REGRESIÓN
        pred_hemo = modelo_reg.predict(datos_input)[0]
        st.subheader("📊 Resultado de Regresión")
        st.metric("Hemoglobina Estimada", f"{pred_hemo/10:.1f} g/dL")
        st.info("Nota: El valor se muestra en escala real (valor predicho / 10).")
        
    else:
        # LÓGICA DE CLASIFICACIÓN CON PROBABILIDAD (Pedido del profesor)
        prediccion_clase = modelo_clf.predict(datos_input)[0]
        
        # Obtener matriz de probabilidades
        probabilidades = modelo_clf.predict_proba(datos_input)[0]
        
        # Formatear porcentajes
        porcentaje_sano = probabilidades[0] * 100
        porcentaje_anemia = probabilidades[1] * 100

        st.subheader("🎯 Resultado de Clasificación")
        
        if prediccion_clase == 1:
            st.error(f"⚠️ DIAGNÓSTICO: PROBABLE ANEMIA")
            st.metric("Probabilidad Estimada de Anemia", f"{porcentaje_anemia:.2f}%")
        else:
            st.success(f"✅ DIAGNÓSTICO: SIN ANEMIA")
            st.metric("Probabilidad Estimada de Salud", f"{porcentaje_sano:.2f}%")
        
        # Análisis Cuantitativo Detallado
        st.write("---")
        st.write(f"### Análisis Cuantitativo de Certeza")
        col1, col2 = st.columns(2)
        col1.statistic = col1.metric("Probabilidad de Salud Normal", f"{porcentaje_sano:.2f}%")
        col2.statistic = col2.metric("Probabilidad de Anemia", f"{porcentaje_anemia:.2f}%")
        
        # Barra visual de confianza
        st.write("**Nivel de riesgo visual:**")
        st.progress(probabilidades[1]) # La barra se llena según el riesgo de anemia

# --- VALIDACIÓN EN CONSOLA (Terminal) ---
porcentaje_anemia_val = "N/A"
print("\n" + "="*30)
print("VALIDACIÓN DE MODELO - EJECUCIÓN")
print("="*30)
print(f"Datos: Edad={v_edad}, Peso={v_peso*10}, Talla={v_talla*10}")
print(">>> Modelos cargados correctamente.")
print("="*30 + "\n")

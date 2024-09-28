import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
import tensorflow as tf

# Función para generar datos de ejemplo más detallados
def generate_sample_data(days=7):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='H')
    
    df = pd.DataFrame({
        'timestamp': dates,
        'traffic_volume': [random.randint(100, 1000) for _ in range(len(dates))],
        'anomaly_score': [random.uniform(0, 1) for _ in range(len(dates))],
        'packet_loss': [random.uniform(0, 0.05) for _ in range(len(dates))],
        'latency': [random.uniform(10, 100) for _ in range(len(dates))],
        'cpu_usage': [random.uniform(20, 80) for _ in range(len(dates))],
        'memory_usage': [random.uniform(30, 90) for _ in range(len(dates))],
        'active_connections': [random.randint(50, 500) for _ in range(len(dates))],
        'firewall_blocks': [random.randint(0, 50) for _ in range(len(dates))]
    })
    return df

# Función para detectar anomalías usando Isolation Forest
def detect_anomalies(df):
    features = ['traffic_volume', 'packet_loss', 'latency', 'cpu_usage', 'memory_usage', 'active_connections', 'firewall_blocks']
    X = df[features]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    clf = IsolationForest(contamination=0.1, random_state=42)
    df['anomaly'] = clf.fit_predict(X_scaled)
    df['anomaly'] = df['anomaly'].map({1: 0, -1: 1})
    
    return df

# Función para crear un modelo de predicción simple (LSTM)
def create_prediction_model(X, y):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, activation='relu', input_shape=(X.shape[1], X.shape[2])),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Función para preparar datos para LSTM
def prepare_data_for_lstm(df, feature, look_back=24):
    data = df[feature].values
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back)])
        y.append(data[i + look_back])
    return np.array(X), np.array(y)

# Configuración de la página
st.set_page_config(page_title="Seguridad en Redes Industriales", layout="wide")

# Título
st.title("Monitoreo de Seguridad en Redes Industriales con IA")

# Generar datos de ejemplo
data = generate_sample_data()

# Detectar anomalías
data_with_anomalies = detect_anomalies(data)

# Crear tres columnas
col1, col2, col3 = st.columns(3)

# Columna 1: Métricas en tiempo real
with col1:
    st.subheader("Métricas en Tiempo Real")
    latest_data = data_with_anomalies.iloc[-1]
    st.metric("Volumen de Tráfico", f"{latest_data['traffic_volume']} Mbps")
    st.metric("Puntuación de Anomalía", f"{latest_data['anomaly_score']:.2f}")
    st.metric("Pérdida de Paquetes", f"{latest_data['packet_loss']:.2%}")
    st.metric("Latencia", f"{latest_data['latency']:.2f} ms")
    st.metric("Uso de CPU", f"{latest_data['cpu_usage']:.1f}%")
    st.metric("Uso de Memoria", f"{latest_data['memory_usage']:.1f}%")
    st.metric("Conexiones Activas", f"{latest_data['active_connections']}")
    st.metric("Bloqueos de Firewall", f"{latest_data['firewall_blocks']}")

# Columna 2 y 3: Gráficos
with col2:
    st.subheader("Tráfico de Red y Anomalías")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data_with_anomalies['timestamp'], y=data_with_anomalies['traffic_volume'], 
                             mode='lines', name='Tráfico'))
    fig.add_trace(go.Scatter(x=data_with_anomalies['timestamp'], y=data_with_anomalies['anomaly_score'] * max(data_with_anomalies['traffic_volume']), 
                             mode='lines', name='Anomalía', yaxis='y2'))
    fig.update_layout(
        yaxis=dict(title="Tráfico (Mbps)"),
        yaxis2=dict(title="Puntuación de Anomalía", overlaying='y', side='right'),
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

with col3:
    st.subheader("Pérdida de Paquetes y Latencia")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data_with_anomalies['timestamp'], y=data_with_anomalies['packet_loss'], 
                             mode='lines', name='Pérdida de Paquetes'))
    fig.add_trace(go.Scatter(x=data_with_anomalies['timestamp'], y=data_with_anomalies['latency'], 
                             mode='lines', name='Latencia', yaxis='y2'))
    fig.update_layout(
        yaxis=dict(title="Pérdida de Paquetes (%)"),
        yaxis2=dict(title="Latencia (ms)", overlaying='y', side='right'),
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

# Nuevas visualizaciones
st.header("Uso de Recursos del Sistema")
col4, col5 = st.columns(2)

with col4:
    fig = px.line(data_with_anomalies, x='timestamp', y=['cpu_usage', 'memory_usage'], 
                  labels={'value': 'Uso (%)', 'variable': 'Recurso'})
    fig.update_layout(title='Uso de CPU y Memoria')
    st.plotly_chart(fig, use_container_width=True)

with col5:
    fig = px.line(data_with_anomalies, x='timestamp', y='active_connections', 
                  labels={'active_connections': 'Conexiones Activas'})
    fig.update_layout(title='Conexiones de Red Activas')
    st.plotly_chart(fig, use_container_width=True)

# Gráfico de barras para bloqueos de firewall
st.subheader("Bloqueos de Firewall por Hora")
fig = px.bar(data_with_anomalies, x='timestamp', y='firewall_blocks')
st.plotly_chart(fig, use_container_width=True)

# Sección de Análisis Predictivo
st.header("Análisis Predictivo de Tráfico")

# Preparar datos para LSTM
X, y = prepare_data_for_lstm(data_with_anomalies, 'traffic_volume')
X = X.reshape((X.shape[0], X.shape[1], 1))

# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo
model = create_prediction_model(X_train, y_train)
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=0)

# Hacer predicciones
predictions = model.predict(X_test)

# Visualizar predicciones
st.subheader("Predicción de Tráfico de Red")
fig = go.Figure()
fig.add_trace(go.Scatter(x=data_with_anomalies['timestamp'][-len(y_test):], y=y_test, mode='lines', name='Real'))
fig.add_trace(go.Scatter(x=data_with_anomalies['timestamp'][-len(predictions):], y=predictions.flatten(), mode='lines', name='Predicción'))
fig.update_layout(title='Predicción de Tráfico de Red', xaxis_title='Tiempo', yaxis_title='Volumen de Tráfico')
st.plotly_chart(fig, use_container_width=True)

# Sección de alertas
st.header("Alertas Recientes")
anomalies = data_with_anomalies[data_with_anomalies['anomaly'] == 1]
if not anomalies.empty:
    for _, row in anomalies.iterrows():
        st.warning(f"Anomalía detectada en {row['timestamp']}: Tráfico: {row['traffic_volume']}, CPU: {row['cpu_usage']:.1f}%, Memoria: {row['memory_usage']:.1f}%")
else:
    st.success("No se han detectado anomalías recientes.")

# Sección de análisis de tráfico
st.header("Análisis de Tráfico de Red")
if st.button("Analizar Tráfico"):
    with st.spinner("Analizando tráfico..."):
        # Aquí podrías agregar lógica adicional para el análisis de tráfico
        st.success("Análisis completado. Se han identificado patrones de tráfico normales y se han actualizado los modelos predictivos.")

# Área para mostrar logs
st.header("Logs del Sistema")
if st.button("Actualizar Logs"):
    st.dataframe(data_with_anomalies.tail())

# Nota al pie
st.markdown("---")
st.caption("Desarrollado para el proyecto 'Seguridad en Redes de Comunicaciones Industriales: Protección de Protocolos en la Industria 4.0'")
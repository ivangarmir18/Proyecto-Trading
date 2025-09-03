"""
dashboard/app.py - Dashboard Streamlit completo con visualizaciones interactivas
Características:
- Visualización en tiempo real de velas y scores
- Señales de trading con stop/target
- Métricas de performance
- Configuración interactiva
- Actualización automática
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import json
from pathlib import Path
import sqlite3
import os

# Configuración de la página
st.set_page_config(
    page_title="Trading Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título y descripción
st.title("📊 Sistema de Trading Inteligente")

# integración no destructiva del widget de watchlist
try:
    from core.storage_postgres import PostgresStorage
    storage = PostgresStorage()  # usa DATABASE_URL si está definido
    from dashboard.watchlist_ui import render_watchlist_ui
    render_watchlist_ui(storage)
except Exception as _e:
    # Si por alguna razón no conecta, seguimos sin romper el dashboard principal.
    st.sidebar.info("Watchlist deshabilitada (configuración requerida).")

st.markdown("""
Dashboard en tiempo real con señales de trading, análisis técnico y predicciones de IA.
""")

# Sidebar para configuración
with st.sidebar:
    st.header("⚙️ Configuración")
    
    # Selección de asset e intervalo
    assets = ["BTCUSDT", "ETHUSDT", "AAPL", "TSLA", "MSFT"]
    intervals = ["5m", "15m", "1h", "4h", "1d"]
    
    selected_asset = st.selectbox("Asset", assets, index=0)
    selected_interval = st.selectbox("Intervalo", intervals, index=2)
    
    # Configuración de visualización
    lookback_days = st.slider("Días a mostrar", 1, 365, 30)
    show_signals = st.checkbox("Mostrar señales", value=True)
    show_indicators = st.checkbox("Mostrar indicadores", value=True)
    
    # Configuración de IA
    ai_enabled = st.checkbox("Habilitar IA", value=True)
    confidence_threshold = st.slider("Umbral de confianza", 0.0, 1.0, 0.6)
    
    # Botones de control
    col1, col2 = st.columns(2)
    with col1:
        refresh_btn = st.button("🔄 Actualizar")
    with col2:
        auto_refresh = st.checkbox("Auto-actualizar", value=True)

# Estilos CSS personalizados
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .signal-buy {
        background-color: #d4edda !important;
        border-left: 4px solid #28a745 !important;
    }
    .signal-sell {
        background-color: #f8d7da !important;
        border-left: 4px solid #dc3545 !important;
    }
    .plot-container {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

class TradingDashboard:
    def __init__(self):
        self.db_path = self.get_db_path()
        self.last_update = None
        
    def get_db_path(self):
        """Obtiene la ruta de la base de datos"""
        env_db_path = os.getenv("DB_PATH")
        if env_db_path and Path(env_db_path).exists():
            return env_db_path
        
        default_paths = [
            "data/db/data.db",
            "data/db/watchlist.db",
            "../data/db/data.db"
        ]
        
        for path in default_paths:
            if Path(path).exists():
                return path
        
        return None
    
    def load_data(self, asset: str, interval: str, lookback_days: int = 30):
        """Carga datos desde la base de datos"""
        if not self.db_path:
            st.error("No se encontró la base de datos")
            return None, None
        
        try:
            end_dt = datetime.now()
            start_dt = end_dt - timedelta(days=lookback_days)
            start_ts = int(start_dt.timestamp() * 1000)
            end_ts = int(end_dt.timestamp() * 1000)
            
            # Conectar a la base de datos
            conn = sqlite3.connect(self.db_path)
            
            # Cargar velas
            candles_query = """
            SELECT ts, open, high, low, close, volume 
            FROM candles 
            WHERE asset = ? AND interval = ? AND ts BETWEEN ? AND ?
            ORDER BY ts
            """
            df_candles = pd.read_sql_query(candles_query, conn, 
                                         params=(asset, interval, start_ts, end_ts))
            
            if not df_candles.empty:
                df_candles['ts'] = pd.to_datetime(df_candles['ts'], unit='ms', utc=True)
                df_candles.set_index('ts', inplace=True)
            
            # Cargar scores
            scores_query = """
            SELECT ts, score, range_min, range_max, stop, target, p_ml, signal_quality
            FROM scores 
            WHERE asset = ? AND interval = ? AND ts BETWEEN ? AND ?
            ORDER BY ts
            """
            df_scores = pd.read_sql_query(scores_query, conn,
                                        params=(asset, interval, start_ts, end_ts))
            
            if not df_scores.empty:
                df_scores['ts'] = pd.to_datetime(df_scores['ts'], unit='ms', utc=True)
                df_scores.set_index('ts', inplace=True)
            
            conn.close()
            
            return df_candles, df_scores
            
        except Exception as e:
            st.error(f"Error cargando datos: {e}")
            return None, None
    
    def create_candlestick_chart(self, df_candles: pd.DataFrame, df_scores: pd.DataFrame = None):
        """Crea gráfico de velas con señales"""
        if df_candles is None or df_candles.empty:
            return None
        
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('Precio y Señales', 'Volumen'),
            row_width=[0.7, 0.3]
        )
        
        # Gráfico de velas
        fig.add_trace(
            go.Candlestick(
                x=df_candles.index,
                open=df_candles['open'],
                high=df_candles['high'],
                low=df_candles['low'],
                close=df_candles['close'],
                name='Precio'
            ),
            row=1, col=1
        )
        
        # Añadir señales si existen
        if df_scores is not None and not df_scores.empty:
            # Merge scores con candles
            df_merged = df_candles.join(df_scores, how='left')
            
            # Señales de compra (score alto)
            buy_signals = df_merged[df_merged['score'] > 0.7]
            if not buy_signals.empty:
                fig.add_trace(
                    go.Scatter(
                        x=buy_signals.index,
                        y=buy_signals['close'],
                        mode='markers',
                        marker=dict(
                            symbol='triangle-up',
                            size=12,
                            color='green',
                            line=dict(width=2, color='darkgreen')
                        ),
                        name='Señal Compra',
                        hovertemplate='<b>Compra</b><br>Precio: %{y:.2f}<br>Score: %{customdata[0]:.2f}<extra></extra>',
                        customdata=buy_signals[['score']].values
                    ),
                    row=1, col=1
                )
            
            # Stop y target levels
            if 'stop' in df_merged.columns and 'target' in df_merged.columns:
                valid_signals = df_merged.dropna(subset=['stop', 'target'])
                for idx, row in valid_signals.iterrows():
                    # Línea de stop
                    fig.add_trace(
                        go.Scatter(
                            x=[idx, idx + timedelta(hours=24)],
                            y=[row['stop'], row['stop']],
                            mode='lines',
                            line=dict(color='red', width=2, dash='dash'),
                            showlegend=False,
                            hoverinfo='skip'
                        ),
                        row=1, col=1
                    )
                    
                    # Línea de target
                    fig.add_trace(
                        go.Scatter(
                            x=[idx, idx + timedelta(hours=24)],
                            y=[row['target'], row['target']],
                            mode='lines',
                            line=dict(color='green', width=2, dash='dash'),
                            showlegend=False,
                            hoverinfo='skip'
                        ),
                        row=1, col=1
                    )
        
        # Gráfico de volumen
        fig.add_trace(
            go.Bar(
                x=df_candles.index,
                y=df_candles['volume'],
                name='Volumen',
                marker_color='rgba(100, 100, 100, 0.5)'
            ),
            row=2, col=1
        )
        
        # Actualizar layout
        fig.update_layout(
            height=800,
            showlegend=True,
            xaxis_rangeslider_visible=False,
            template='plotly_white',
            hovermode='x unified'
        )
        
        fig.update_yaxes(title_text="Precio", row=1, col=1)
        fig.update_yaxes(title_text="Volumen", row=2, col=1)
        fig.update_xaxes(title_text="Fecha", row=2, col=1)
        
        return fig
    
    def create_metrics_panel(self, df_candles: pd.DataFrame, df_scores: pd.DataFrame):
        """Panel de métricas y KPIs"""
        if df_candles is None or df_candles.empty:
            return
        
        # Calcular métricas básicas
        current_price = df_candles['close'].iloc[-1]
        price_change = current_price - df_candles['close'].iloc[-2] if len(df_candles) > 1 else 0
        price_change_pct = (price_change / df_candles['close'].iloc[-2] * 100) if len(df_candles) > 1 else 0
        
        # Métricas de volatilidad
        returns = df_candles['close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) * 100  # Volatilidad anualizada
        
        # Métricas de volumen
        avg_volume = df_candles['volume'].mean()
        current_volume = df_candles['volume'].iloc[-1]
        
        # Métricas de scores si disponibles
        if df_scores is not None and not df_scores.empty:
            current_score = df_scores['score'].iloc[-1] if 'score' in df_scores.columns else 0
            avg_score = df_scores['score'].mean() if 'score' in df_scores.columns else 0
            signal_quality = df_scores['signal_quality'].iloc[-1] if 'signal_quality' in df_scores.columns else 0
        else:
            current_score = avg_score = signal_quality = 0
        
        # Mostrar métricas
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Precio Actual",
                f"${current_price:.2f}",
                f"{price_change_pct:+.2f}%"
            )
        
        with col2:
            st.metric(
                "Volatilidad Anual",
                f"{volatility:.1f}%"
            )
        
        with col3:
            st.metric(
                "Score Actual",
                f"{current_score:.2f}",
                f"Avg: {avg_score:.2f}"
            )
        
        with col4:
            st.metric(
                "Calidad Señal",
                f"{signal_quality:.2f}"
            )
    
    def create_signal_details(self, df_scores: pd.DataFrame):
        """Detalles de las señales actuales"""
        if df_scores is None or df_scores.empty:
            st.info("No hay señales disponibles")
            return
        
        latest_signal = df_scores.iloc[-1]
        
        # Determinar tipo de señal
        score = latest_signal.get('score', 0)
        if score >= 0.7:
            signal_type = "🟢 COMPRA"
            signal_class = "signal-buy"
        elif score >= 0.5:
            signal_type = "🟡 NEUTRAL"
            signal_class = ""
        else:
            signal_type = "🔻 EVITAR"
            signal_class = "signal-sell"
        
        # Mostrar detalles de la señal
        st.markdown(f"""
        <div class="metric-card {signal_class}">
            <h3>Última Señal: {signal_type}</h3>
            <p><strong>Score:</strong> {score:.2f}</p>
            <p><strong>Confianza IA:</strong> {latest_signal.get('p_ml', 0):.2f}</p>
            <p><strong>Stop:</strong> ${latest_signal.get('stop', 0):.2f}</p>
            <p><strong>Target:</strong> ${latest_signal.get('target', 0):.2f}</p>
            <p><strong>Calidad:</strong> {latest_signal.get('signal_quality', 0):.2f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    def create_performance_chart(self, df_candles: pd.DataFrame, df_scores: pd.DataFrame):
        """Gráfico de performance y scores"""
        if df_candles is None or df_candles.empty:
            return None
        
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=('Performance', 'Scores'),
            row_width=[0.6, 0.4]
        )
        
        # Precio normalizado
        base_price = df_candles['close'].iloc[0]
        normalized_price = (df_candles['close'] / base_price - 1) * 100
        
        fig.add_trace(
            go.Scatter(
                x=df_candles.index,
                y=normalized_price,
                name='Performance',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # Scores si disponibles
        if df_scores is not None and not df_scores.empty:
            # Merge para alinear timestamps
            df_merged = df_candles.join(df_scores[['score']], how='left')
            df_merged['score'] = df_merged['score'].fillna(0.5)
            
            fig.add_trace(
                go.Scatter(
                    x=df_merged.index,
                    y=df_merged['score'] * 100,  # Escalar a 0-100
                    name='Score',
                    line=dict(color='orange', width=2)
                ),
                row=2, col=1
            )
        
        fig.update_layout(
            height=600,
            showlegend=True,
            template='plotly_white'
        )
        
        fig.update_yaxes(title_text="Performance (%)", row=1, col=1)
        fig.update_yaxes(title_text="Score (%)", row=2, col=1)
        fig.update_xaxes(title_text="Fecha", row=2, col=1)
        
        return fig
    
    def run(self):
        """Ejecutar el dashboard"""
        dashboard = TradingDashboard()
        
        # Cargar datos iniciales
        df_candles, df_scores = dashboard.load_data(
            selected_asset, selected_interval, lookback_days
        )
        
        if df_candles is None:
            st.error("No se pudieron cargar los datos")
            return
        
        # Panel de métricas
        dashboard.create_metrics_panel(df_candles, df_scores)
        
        # Layout principal
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📈 Gráfico de Precio y Señales")
            fig = dashboard.create_candlestick_chart(df_candles, df_scores)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📊 Detalles de Señal")
            dashboard.create_signal_details(df_scores)
            
            st.subheader("⚡ Acciones Rápidas")
            if st.button("🔄 Forzar Actualización", type="primary"):
                st.rerun()
            
            if st.button("📊 Ver Métricas Detalladas"):
                st.session_state.show_metrics = True
            
            if st.button("🤖 Entrenar Modelo IA"):
                # Aquí iría la llamada al entrenamiento de IA
                st.info("Función de entrenamiento en desarrollo")
        
        # Gráfico de performance
        st.subheader("📊 Performance y Scores")
        perf_fig = dashboard.create_performance_chart(df_candles, df_scores)
        if perf_fig:
            st.plotly_chart(perf_fig, use_container_width=True)
        
        # Auto-refresh
        if auto_refresh:
            refresh_interval = 300  # 5 minutos
            if dashboard.last_update is None or time.time() - dashboard.last_update > refresh_interval:
                time.sleep(2)
                st.rerun()

# Ejecutar el dashboard
if __name__ == "__main__":
    dashboard = TradingDashboard()
    dashboard.run()
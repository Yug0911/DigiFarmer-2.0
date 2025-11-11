# =========================
# app.py (DigiFarmer 2.0+)
# =========================
import os
import time
import joblib
import pydeck as pdk
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from preprocessing import predict_crop
from api_utils import get_location, get_soil_data, get_weather_data, get_soil_moisture

# =========================
# THEME (agri palette)
# =========================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Amatic+SC:wght@400;700&family=Poppins:wght@300;400;600&display=swap');
    body { background: linear-gradient(135deg, #1a1a1a 0%, #000000 100%); font-family: 'Poppins', sans-serif; color: #ffffff; }
    .stApp { background: linear-gradient(135deg, #1a1a1a 0%, #000000 100%); }
    .title { font-family: 'Amatic SC', cursive; font-size: 3em; text-align: center; color: #ffd700;
             text-shadow: 2px 2px 4px rgba(0,0,0,0.5); }
    .subtitle { text-align: center; font-style: italic; color: #f4f4f4; margin-bottom: 16px; }
    .input-card { background: rgba(255, 255, 255, 0.05); border-radius: 14px; padding: 12px; margin: 8px 0;
                  border: 1px solid rgba(255, 255, 255, 0.25); }
    .result-card { background: rgba(255,255,255,0.1); border-radius: 18px; padding: 18px; margin: 16px 0;
                   border: 2px solid #ffd700; }
    h1,h2,h3 { color: #ffd700 !important; font-family: 'Amatic SC', cursive; }
    .footer { text-align: center; margin-top: 30px; color: #eaeaea; font-size: 0.9em; }
</style>
""", unsafe_allow_html=True)

# =========================
# CONSTANTS / HELPERS
# =========================
REQUIRED_NUMERICAL = ['N', 'P', 'K', 'Temperature', 'Humidity', 'Soil_pH', 'Wind_Speed']
REQUIRED_CATEGORICAL = ['Soil_Type', 'Crop_Type']
REQUIRED_ALL = REQUIRED_NUMERICAL + REQUIRED_CATEGORICAL

@st.cache_data
def load_viz_data(csv_path: str):
    df = pd.read_csv(csv_path)

    # Validate
    missing = [c for c in REQUIRED_ALL if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}. Present: {list(df.columns)}")

    # Fill NA
    for col in REQUIRED_NUMERICAL:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mean())
    for col in REQUIRED_CATEGORICAL:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])

    # Encode soil for visuals only
    le_local = LabelEncoder()
    df['soil_type_encoded'] = le_local.fit_transform(df['Soil_Type'])

    # Scale numericals for visuals
    scaler_local = StandardScaler()
    df_scaled = df.copy()
    df_scaled[REQUIRED_NUMERICAL] = scaler_local.fit_transform(df_scaled[REQUIRED_NUMERICAL])

    return df, df_scaled

def kpi_row(params):
    c1, c2, c3 = st.columns(3)
    c1.metric("Nitrogen (N)", f"{params['N']:.1f}")
    c2.metric("Phosphorus (P)", f"{params['P']:.1f}")
    c3.metric("Potassium (K)", f"{params['K']:.1f}")
    c4, c5, c6, c7 = st.columns(4)
    c4.metric("Temp (°C)", f"{params['Temperature']:.1f}")
    c5.metric("Humidity (%)", f"{params['Humidity']:.0f}")
    c6.metric("Soil pH", f"{params['Soil_pH']:.2f}")
    c7.metric("Wind (km/h)", f"{params['Wind_Speed']:.1f}")

def pack_inputs(n, p, k, temperature, humidity, ph, rainfall, moisture, windspeed, soil_type):
    return dict(N=n, P=p, K=k, Temperature=temperature, Humidity=humidity,
                Soil_pH=ph, Rainfall=rainfall, Moisture=moisture,
                Wind_Speed=windspeed, Soil_Type=soil_type)


# =========================
# TITLE + SUBTITLE
# =========================
st.markdown('<h1 class="title">🌾 DigiFarmer • Crop Oracle</h1>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Smart, explainable crop recommendations with beautiful agri dashboards.</div>', unsafe_allow_html=True)

# Soil type options from saved encoder
le = joblib.load('label_encoder.pkl')
soil_types = list(le.classes_)  # ensure list for selectbox

# =========================
# TABS
# =========================
tab_reco, tab_insights, tab_field, tab_guide = st.tabs(
    ["🔮 Recommendation", "📈 Insights", "🗺️ Field Dashboard", "🌿 Crop Guide"]
)

# =========================
# RECOMMENDATION TAB
# =========================
with tab_reco:
    st.subheader("🌍 Location & Environmental Data")
    col_location, col_fetch = st.columns([2, 1])

    with col_location:
        use_location = st.checkbox("Use my current location (auto fetch)", value=False)
        if use_location and st.button("📍 Fetch location & data"):
            with st.spinner("Detecting location & fetching soil/weather..."):
                location = get_location()
                if location:
                    st.session_state.location = location
                    st.success(f"Location: {location['city']}, {location['country']}")
                    soil_data = get_soil_data(location['lat'], location['lon'])
                    if soil_data:
                        st.session_state.soil_data = soil_data
                        st.info("Soil data fetched.")
                    weather_data = get_weather_data(location['lat'], location['lon'])
                    if weather_data:
                        st.session_state.weather_data = weather_data
                        st.info("Weather data fetched.")
                    moisture_data = get_soil_moisture(location['lat'], location['lon'])
                    if moisture_data.get('moisture'):
                        st.session_state.moisture_data = moisture_data
                else:
                    st.error("Unable to detect location. Enter values manually.")

    with col_fetch:
        if 'location' in st.session_state:
            st.markdown("**Current Location**")
            st.write(f"📍 {st.session_state.location['city']}, {st.session_state.location['country']}")
            st.write(f"Lat: {st.session_state.location['lat']:.4f}, Lon: {st.session_state.location['lon']:.4f}")

    # Demo presets
    st.subheader("🧪 Quick Demo Scenarios")
    demo = st.selectbox("Pick a preset", ["None", "Humid rice-like", "Wheat-like", "Dry & hot (groundnut)"])
    presets = {
        "Humid rice-like": dict(N=90, P=42, K=43, Temperature=29.5, Humidity=82, Soil_pH=6.5, Rainfall=220, Moisture=35, Wind_Speed=10, Soil_Type=soil_types[0] if soil_types else "Loamy"),
        "Wheat-like": dict(N=35, P=18, K=22, Temperature=19.0, Humidity=55, Soil_pH=7.2, Rainfall=80, Moisture=18, Wind_Speed=6, Soil_Type=soil_types[min(1,len(soil_types)-1)] if soil_types else "Clay"),
        "Dry & hot (groundnut)": dict(N=25, P=12, K=18, Temperature=34.0, Humidity=45, Soil_pH=7.8, Rainfall=60, Moisture=15, Wind_Speed=14, Soil_Type=soil_types[min(2,len(soil_types)-1)] if soil_types else "Sandy"),
    }

    st.subheader("🧪 Inputs")
    col1, col2 = st.columns(2)

    # Defaults (from APIs if present)
    default_n = st.session_state.get('soil_data', {}).get('organic_carbon', 90.0) if 'soil_data' in st.session_state else 90.0
    default_p = 42.0
    default_k = 43.0
    default_temperature = st.session_state.get('weather_data', {}).get('temperature', 24.0) if 'weather_data' in st.session_state else 24.0
    default_humidity = st.session_state.get('weather_data', {}).get('humidity', 70.0) if 'weather_data' in st.session_state else 70.0
    default_ph = st.session_state.get('soil_data', {}).get('ph', 6.5) if 'soil_data' in st.session_state else 6.5
    default_rainfall = st.session_state.get('weather_data', {}).get('rainfall', 140.0) if 'weather_data' in st.session_state else 140.0
    default_moisture = st.session_state.get('moisture_data', {}).get('moisture', 30.0) if st.session_state.get('moisture_data', {}).get('moisture') else 30.0
    default_windspeed = st.session_state.get('weather_data', {}).get('windspeed', 8.0) if 'weather_data' in st.session_state else 8.0

    with col1:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        n = st.number_input("🌱 Nitrogen (N)", min_value=0.0, key="N", value=float(presets.get(demo, {}).get('N', default_n)) if demo != "None" else float(default_n))
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        p = st.number_input("💎 Phosphorus (P)", min_value=0.0, key="P", value=float(presets.get(demo, {}).get('P', default_p)) if demo != "None" else float(default_p))
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        k = st.number_input("⚡ Potassium (K)", min_value=0.0, key="K", value=float(presets.get(demo, {}).get('K', default_k)) if demo != "None" else float(default_k))
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        temperature = st.number_input("🌡️ Temperature (°C)", min_value=0.0, key="Temperature", value=float(presets.get(demo, {}).get('Temperature', default_temperature)) if demo != "None" else float(default_temperature))
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        humidity = st.number_input("💧 Humidity (%)", min_value=0.0, max_value=100.0, key="Humidity", value=float(presets.get(demo, {}).get('Humidity', default_humidity)) if demo != "None" else float(default_humidity))
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        ph = st.number_input("🧪 Soil pH", min_value=0.0, max_value=14.0, key="Soil_pH", value=float(presets.get(demo, {}).get('Soil_pH', default_ph)) if demo != "None" else float(default_ph))
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        rainfall = st.number_input("🌧️ Rainfall (mm)", min_value=0.0, key="Rainfall", value=float(presets.get(demo, {}).get('Rainfall', default_rainfall)) if demo != "None" else float(default_rainfall))
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        moisture = st.number_input("💦 Moisture (%)", min_value=0.0, max_value=100.0, key="Moisture", value=float(presets.get(demo, {}).get('Moisture', default_moisture)) if demo != "None" else float(default_moisture))
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        windspeed = st.number_input("🌬️ Wind Speed (km/h)", min_value=0.0, key="Wind_Speed", value=float(presets.get(demo, {}).get('Wind_Speed', default_windspeed)) if demo != "None" else float(default_windspeed))
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        default_soil = presets.get(demo, {}).get('Soil_Type', soil_types[0] if soil_types else "")
        index = soil_types.index(default_soil) if default_soil in soil_types else 0
        soil_type = st.selectbox("🏔️ Soil Type", soil_types, key="Soil_Type", index=index)
        st.markdown('</div>', unsafe_allow_html=True)

    # Predict
    if st.button("🔮 Reveal the Magical Crop"):
        try:
            with st.spinner("Consulting the agri spirits..."):
                time.sleep(0.8)
                rec = predict_crop(n, p, k, temperature, humidity, ph, rainfall, moisture, soil_type, windspeed)

            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown(f'<h2 style="text-align:center;color:#ffd700;font-size:2.3em;">✨ {rec.upper()} ✨</h2>', unsafe_allow_html=True)
            st.markdown('<p style="text-align:center;">Recommendation generated by ML model.</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            params = pack_inputs(n, p, k, temperature, humidity, ph, rainfall, moisture, windspeed, soil_type)
            kpi_row(params)
            st.json(params)

            # Save scenario
            if "scenarios" not in st.session_state:
                st.session_state.scenarios = {}
            st.text_input("Name this scenario", key="scenario_name", placeholder="e.g., Kheda winter plot")
            if st.button("💾 Save this scenario"):
                name = st.session_state.get("scenario_name", "").strip()
                if name:
                    st.session_state.scenarios[name] = params
                    st.success(f"Saved: {name}")
                else:
                    st.warning("Please provide a scenario name.")
        except Exception as e:
            with st.expander("Show error details"):
                st.exception(e)
            st.error("Prediction failed. Ensure artifacts/CSV exist & preprocessing ran once.")

# =========================
# INSIGHTS TAB
# =========================
with tab_insights:
    st.subheader("🥇 What influenced the recommendation?")
    try:
        model = joblib.load('best_crop_model.pkl')
        if hasattr(model, "feature_importances_"):
            feature_cols = ['N','P','K','Temperature','Humidity','Soil_pH','Wind_Speed',
                            'soil_type_encoded','n_p_ratio','n_k_ratio','p_k_ratio','temp_humidity','wind_temp']
            importances = model.feature_importances_
            order = np.argsort(importances)[::-1]
            top = min(10, len(importances))
            idx = order[:top]
            fig, ax = plt.subplots(figsize=(7,4))
            ax.barh([feature_cols[i] for i in idx][::-1], importances[idx][::-1])
            ax.set_xlabel("Importance"); ax.set_title("Top Feature Importances")
            st.pyplot(fig)
        else:
            st.info("Current model does not expose feature importances.")
    except Exception as e:
        st.warning(f"Could not load model insights: {e}")

    st.divider()
    st.subheader("📊 Explore Dataset Patterns")
    plot_options = [
        "🌱 Distribution of Nitrogen (N)",
        "💎 Distribution of Phosphorus (P)",
        "⚡ Distribution of Potassium (K)",
        "🌡️ Distribution of Temperature",
        "💧 Distribution of Humidity",
        "🧪 Distribution of Soil pH",
        "🌬️ Distribution of Wind Speed",
        "🏔️ Distribution of Soil Types",
        "📈 Correlation Heatmap",
        "📊 Boxplot: Nitrogen by Crop",
        "📊 Boxplot: Phosphorus by Crop",
        "📊 Boxplot: Potassium by Crop",
        "📊 Boxplot: Temperature by Crop",
        "📊 Boxplot: Humidity by Crop",
        "📊 Boxplot: Soil pH by Crop",
        "📊 Boxplot: Wind Speed by Crop"
    ]
    selected_plot = st.selectbox("Choose a visualization:", ["Select a plot..."] + plot_options, key="plot_selector")
    if selected_plot != "Select a plot...":
        try:
            df, df_scaled = load_viz_data('crop_yield_dataset.csv')
            fig, ax = plt.subplots(figsize=(10,6))

            # Distributions
            if "Distribution of Nitrogen" in selected_plot:
                sns.histplot(df_scaled['N'], kde=True, ax=ax); ax.set_title('Distribution of Nitrogen (N)')
            elif "Distribution of Phosphorus" in selected_plot:
                sns.histplot(df_scaled['P'], kde=True, ax=ax); ax.set_title('Distribution of Phosphorus (P)')
            elif "Distribution of Potassium" in selected_plot:
                sns.histplot(df_scaled['K'], kde=True, ax=ax); ax.set_title('Distribution of Potassium (K)')
            elif "Distribution of Temperature" in selected_plot:
                sns.histplot(df_scaled['Temperature'], kde=True, ax=ax); ax.set_title('Distribution of Temperature')
            elif "Distribution of Humidity" in selected_plot:
                sns.histplot(df_scaled['Humidity'], kde=True, ax=ax); ax.set_title('Distribution of Humidity')
            elif "Distribution of Soil pH" in selected_plot:
                sns.histplot(df_scaled['Soil_pH'], kde=True, ax=ax); ax.set_title('Distribution of Soil pH')
            elif "Distribution of Wind Speed" in selected_plot:
                sns.histplot(df_scaled['Wind_Speed'], kde=True, ax=ax); ax.set_title('Distribution of Wind Speed')
            elif "Distribution of Soil Types" in selected_plot:
                sns.histplot(df['soil_type_encoded'], kde=True, ax=ax); ax.set_title('Distribution of Soil Types')

            # Correlation
            elif "Correlation Heatmap" in selected_plot:
                plt.close(fig)
                fig, ax = plt.subplots(figsize=(12,10))
                corr_matrix = df_scaled[REQUIRED_NUMERICAL + ['soil_type_encoded']].corr()
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
                ax.set_title('Correlation Heatmap')

            # Boxplots
            elif "Boxplot:" in selected_plot:
                plt.close(fig)
                fig, ax = plt.subplots(figsize=(12,8))
                if "Nitrogen by Crop" in selected_plot:
                    sns.boxplot(x='Crop_Type', y='N', data=df_scaled, ax=ax); ax.set_title('Nitrogen by Crop')
                elif "Phosphorus by Crop" in selected_plot:
                    sns.boxplot(x='Crop_Type', y='P', data=df_scaled, ax=ax); ax.set_title('Phosphorus by Crop')
                elif "Potassium by Crop" in selected_plot:
                    sns.boxplot(x='Crop_Type', y='K', data=df_scaled, ax=ax); ax.set_title('Potassium by Crop')
                elif "Temperature by Crop" in selected_plot:
                    sns.boxplot(x='Crop_Type', y='Temperature', data=df_scaled, ax=ax); ax.set_title('Temperature by Crop')
                elif "Humidity by Crop" in selected_plot:
                    sns.boxplot(x='Crop_Type', y='Humidity', data=df_scaled, ax=ax); ax.set_title('Humidity by Crop')
                elif "Soil pH by Crop" in selected_plot:
                    sns.boxplot(x='Crop_Type', y='Soil_pH', data=df_scaled, ax=ax); ax.set_title('Soil pH by Crop')
                elif "Wind Speed by Crop" in selected_plot:
                    sns.boxplot(x='Crop_Type', y='Wind_Speed', data=df_scaled, ax=ax); ax.set_title('Wind Speed by Crop')
                ax.tick_params(axis='x', rotation=45)

            st.pyplot(fig); plt.close(fig)
        except FileNotFoundError:
            st.error("CSV not found. Place 'crop_yield_dataset.csv' next to app.py.")
        except Exception as e:
            with st.expander("Show error details"):
                st.exception(e)
            st.error("Failed to render the plot.")

# =========================
# FIELD DASHBOARD TAB
# =========================
with tab_field:
    st.subheader("🗺️ Field Location")
    if 'location' in st.session_state:
        loc = st.session_state.location
        view_state = pdk.ViewState(latitude=loc['lat'], longitude=loc['lon'], zoom=8, pitch=0)
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=[{"lat": loc['lat'], "lon": loc['lon']}],
            get_position='[lon, lat]', get_radius=5000, pickable=True,
        )
        st.pydeck_chart(pdk.Deck(map_style=None, initial_view_state=view_state, layers=[layer]))
    else:
        st.info("Enable location in the Recommendation tab and fetch data to see your field on the map.")

    st.divider()
    st.subheader("🆚 Compare Saved Scenarios")
    if "scenarios" in st.session_state and len(st.session_state.scenarios) >= 2:
        names = list(st.session_state.scenarios.keys())
        left = st.selectbox("Left scenario", names, key="cmp_left")
        right = st.selectbox("Right scenario", [n for n in names if n != st.session_state.cmp_left], key="cmp_right")
        if left and right:
            lvals = st.session_state.scenarios[left]
            rvals = st.session_state.scenarios[right]
            c1, c2 = st.columns(2)
            c1.write(pd.DataFrame([lvals], index=[left]).T)
            c2.write(pd.DataFrame([rvals], index=[right]).T)
    else:
        st.info("Save at least two scenarios in the Recommendation tab to compare.")

# =========================
# CROP GUIDE TAB
# =========================
with tab_guide:
    st.subheader("🌿 Quick Crop Guide (illustrative)")
    st.caption("Ranges are indicative; consult local agronomy for precise recommendations.")
    guide = {
        "Rice":      {"Temp": (20, 35), "Humidity": (60, 95), "pH": (5.5, 7.0), "Rain": (150, 300)},
        "Wheat":     {"Temp": (15, 25), "Humidity": (40, 70), "pH": (6.0, 7.5), "Rain": (50, 120)},
        "Maize":     {"Temp": (18, 30), "Humidity": (40, 80), "pH": (5.8, 7.2), "Rain": (60, 150)},
        "Groundnut": {"Temp": (25, 35), "Humidity": (30, 60), "pH": (6.0, 7.5), "Rain": (50, 100)},
        "Cotton":    {"Temp": (21, 32), "Humidity": (30, 60), "pH": (5.8, 8.0), "Rain": (50, 100)}
    }
    def chip(ok):
        color = "#33cc66" if ok else "#ffaa33"
        return f"background:{color};padding:2px 8px;border-radius:12px;color:white;font-size:12px;margin-right:6px;"
    for crop, rg in guide.items():
        st.markdown(f"**{crop}**")
        # Use current inputs if available, else mid-range defaults
        t = st.session_state.get("Temperature", 25.0)
        h = st.session_state.get("Humidity", 60.0)
        pH = st.session_state.get("Soil_pH", 6.5)
        rain = st.session_state.get("Rainfall", 120.0)
        t_ok = rg["Temp"][0] <= t <= rg["Temp"][1]
        h_ok = rg["Humidity"][0] <= h <= rg["Humidity"][1]
        p_ok = rg["pH"][0] <= pH <= rg["pH"][1]
        r_ok = rg["Rain"][0] <= rain <= rg["Rain"][1]
        st.markdown(
            f"""
            <div>
                <span style="{chip(t_ok)}">Temp {rg['Temp'][0]}–{rg['Temp'][1]}°C</span>
                <span style="{chip(h_ok)}">Humidity {rg['Humidity'][0]}–{rg['Humidity'][1]}%</span>
                <span style="{chip(p_ok)}">pH {rg['pH'][0]}–{rg['pH'][1]}</span>
                <span style="{chip(r_ok)}">Rain {rg['Rain'][0]}–{rg['Rain'][1]} mm</span>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.divider()

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown('<div class="footer">🌟 Built with love for growers. May your harvests be bountiful! 🌟</div>', unsafe_allow_html=True)

import streamlit as st
import pandas as pd
import joblib
from preprocessing import predict_crop
from api_utils import get_location, get_soil_data, get_weather_data, get_soil_moisture
import streamlit.components.v1 as components

# Load the label encoders to get soil type options
le = joblib.load('label_encoder.pkl')
soil_types = le.classes_

# Custom CSS for unique, magical garden theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Amatic+SC:wght@400;700&family=Poppins:wght@300;400;600&display=swap');

    body {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Poppins', sans-serif;
        color: #ffffff;
    }

    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }

    .css-1d391kg {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 20px;
        margin: 10px 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }

    .title {
        font-family: 'Amatic SC', cursive;
        font-size: 3em;
        text-align: center;
        color: #ffd700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        animation: glow 2s ease-in-out infinite alternate;
    }

    @keyframes glow {
        from { text-shadow: 2px 2px 4px rgba(0,0,0,0.5); }
        to { text-shadow: 2px 2px 4px rgba(255,215,0,0.8); }
    }

    .subtitle {
        text-align: center;
        font-style: italic;
        color: #e0e0e0;
        margin-bottom: 30px;
    }

    .input-card {
        background: rgba(255, 255, 255, 0.15);
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
        border: 1px solid rgba(255, 255, 255, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    .input-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
    }

    .stButton>button {
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 12px 30px;
        font-size: 18px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        animation: pulse 2s infinite;
    }

    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
    }

    @keyframes pulse {
        0% { box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2); }
        50% { box-shadow: 0 4px 15px rgba(255, 107, 107, 0.4); }
        100% { box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2); }
    }

    .result-card {
        background: rgba(255, 255, 255, 0.2);
        border-radius: 20px;
        padding: 20px;
        margin: 20px 0;
        border: 2px solid #ffd700;
        animation: fadeIn 1s ease-in;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .stSuccess {
        background: rgba(76, 175, 80, 0.8) !important;
        color: white !important;
        border-radius: 15px;
        padding: 15px;
        font-size: 1.2em;
        text-align: center;
    }

    .stInfo {
        background: rgba(33, 150, 243, 0.8) !important;
        color: white !important;
        border-radius: 15px;
        padding: 15px;
    }

    .stJson {
        background: rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px;
        padding: 10px;
    }

    h1, h2, h3 {
        color: #ffd700 !important;
        font-family: 'Amatic SC', cursive;
    }

    .footer {
        text-align: center;
        margin-top: 50px;
        color: #e0e0e0;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="title">🌱 Magical Crop Oracle 🌱</h1>', unsafe_allow_html=True)

st.markdown("""
<div class="subtitle">
    <em>Whisper your soil's secrets, and let the ancient wisdom of nature reveal the perfect crop to nurture your land.</em>
</div>
""", unsafe_allow_html=True)

# Geolocation and API Data Fetching Section
st.markdown("### 🌍 Location & Environmental Data 🌍")

col_location, col_fetch = st.columns([2, 1])

with col_location:
    use_location = st.checkbox("🌐 Use my current location for automatic data fetching", value=False)

    if use_location:
        if st.button("📍 Get My Location & Fetch Data"):
            with st.spinner("🔍 Detecting your location and fetching environmental data..."):
                location = get_location()
                if location:
                    st.session_state.location = location
                    st.success(f"📍 Location detected: {location['city']}, {location['country']}")

                    # Fetch soil data
                    soil_data = get_soil_data(location['lat'], location['lon'])
                    if soil_data:
                        st.session_state.soil_data = soil_data
                        st.info("🌱 Soil data fetched successfully!")

                    # Fetch weather data
                    weather_data = get_weather_data(location['lat'], location['lon'])
                    if weather_data:
                        st.session_state.weather_data = weather_data
                        st.info("🌤️ Weather data fetched successfully!")

                    # Optional soil moisture
                    moisture_data = get_soil_moisture(location['lat'], location['lon'])
                    if moisture_data.get('moisture'):
                        st.session_state.moisture_data = moisture_data
                else:
                    st.error("❌ Unable to detect location. Please enter data manually.")

with col_fetch:
    if 'location' in st.session_state:
        st.markdown("**Current Location:**")
        st.write(f"📍 {st.session_state.location['city']}, {st.session_state.location['country']}")
        st.write(f"Lat: {st.session_state.location['lat']:.4f}, Lon: {st.session_state.location['lon']:.4f}")

st.markdown("### 🌿 Enchant Your Soil's Essence 🌿")

col1, col2 = st.columns(2)

# Auto-fill from API data if available
default_n = st.session_state.get('soil_data', {}).get('organic_carbon', 90.0) if 'soil_data' in st.session_state else 90.0
default_p = 42.0  # Phosphorus not directly from API
default_k = 43.0  # Potassium not directly from API
default_temperature = st.session_state.get('weather_data', {}).get('temperature', 20.87) if 'weather_data' in st.session_state else 20.87
default_humidity = st.session_state.get('weather_data', {}).get('humidity', 82.00) if 'weather_data' in st.session_state else 82.00
default_ph = st.session_state.get('soil_data', {}).get('ph', 6.50) if 'soil_data' in st.session_state else 6.50
default_rainfall = st.session_state.get('weather_data', {}).get('rainfall', 202.93) if 'weather_data' in st.session_state else 202.93
default_moisture = st.session_state.get('moisture_data', {}).get('moisture', 29.44) if 'moisture_data' in st.session_state and st.session_state['moisture_data'].get('moisture') else 29.44
default_windspeed = st.session_state.get('weather_data', {}).get('windspeed', 10.10) if 'weather_data' in st.session_state else 10.10

with col1:
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    n = st.number_input("🌱 Nitrogen (N) Essence", min_value=0.0, value=float(default_n), help="The life-giving nitrogen content (Organic Carbon from API)" if 'soil_data' in st.session_state else "The life-giving nitrogen content")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    p = st.number_input("💎 Phosphorus (P) Crystals", min_value=0.0, value=42.0, help="The root-strengthening phosphorus")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    k = st.number_input("⚡ Potassium (K) Energy", min_value=0.0, value=43.0, help="The vitality-boosting potassium")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    temperature = st.number_input("🌡️ Temperature (°C)", min_value=0.0, value=float(default_temperature), help="The warmth of the earth (from weather API)" if 'weather_data' in st.session_state else "The warmth of the earth")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    humidity = st.number_input("💧 Humidity (%)", min_value=0.0, max_value=100.0, value=float(default_humidity), help="The moisture in the air (from weather API)" if 'weather_data' in st.session_state else "The moisture in the air")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    ph = st.number_input("🧪 pH Balance", min_value=0.0, max_value=14.0, value=float(default_ph), help="The soil's acidic/alkaline harmony (from soil API)" if 'soil_data' in st.session_state else "The soil's acidic/alkaline harmony")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    rainfall = st.number_input("🌧️ Rainfall (mm)", min_value=0.0, value=float(default_rainfall), help="The tears of the sky (from weather API)" if 'weather_data' in st.session_state else "The tears of the sky")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    moisture = st.number_input("💦 Moisture (%)", min_value=0.0, max_value=100.0, value=float(default_moisture), help="The soil's hidden waters (from moisture API)" if 'moisture_data' in st.session_state and st.session_state['moisture_data'].get('moisture') else "The soil's hidden waters")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    windspeed = st.number_input("🌬️ Wind Speed (km/h)", min_value=0.0, value=float(default_windspeed), help="The breath of the winds (from weather API)" if 'weather_data' in st.session_state else "The breath of the winds")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    soil_type = st.selectbox("🏔️ Soil Type", soil_types, help="The ancient earth beneath your feet")
    st.markdown('</div>', unsafe_allow_html=True)

if st.button("🔮 Reveal the Magical Crop 🔮"):
    try:
        with st.spinner("🌟 Consulting the ancient spirits of agriculture... 🌟"):
            import time
            time.sleep(2)  # Simulate mystical processing time
            recommended_crop = predict_crop(n, p, k, temperature, humidity, ph, rainfall, moisture, soil_type, windspeed)

        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown(f'<h2 style="text-align: center; color: #ffd700; font-size: 2.5em;">✨ {recommended_crop.upper()} ✨</h2>', unsafe_allow_html=True)
        st.markdown('<p style="text-align: center; font-size: 1.2em;">has been chosen by the mystical forces of nature!</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("### 📜 Your Enchanted Parameters 📜")
        params = {
            "🌱 Nitrogen (N) Essence": n,
            "💎 Phosphorus (P) Crystals": p,
            "⚡ Potassium (K) Energy": k,
            "🌡️ Temperature (°C)": temperature,
            "💧 Humidity (%)": humidity,
            "🧪 pH Balance": ph,
            "🌧️ Rainfall (mm)": rainfall,
            "💦 Moisture (%)": moisture,
            "🌬️ Wind Speed (km/h)": windspeed,
            "🏔️ Soil Type": soil_type
        }

        # Add API data sources if available
        if 'location' in st.session_state:
            params["📍 Location"] = f"{st.session_state.location['city']}, {st.session_state.location['country']}"
            params["Coordinates"] = f"Lat: {st.session_state.location['lat']:.4f}, Lon: {st.session_state.location['lon']:.4f}"

        if 'soil_data' in st.session_state:
            params["🌱 Soil Data Source"] = "SoilGrids API"
            if 'cec' in st.session_state.soil_data:
                params["🧪 Cation Exchange Capacity"] = st.session_state.soil_data['cec']

        if 'weather_data' in st.session_state:
            params["🌤️ Weather Data Source"] = "OpenWeatherMap API"

        st.json(params)

        st.info("🌙 This divine recommendation emerges from the wisdom of machine learning alchemy. Seek counsel from earthly sages for your final harvest decisions. 🌙")

        # Plot visualization section
        st.markdown("### 📊 Explore the Mystical Data Patterns 📊")

        plot_options = [
            "🌱 Distribution of Nitrogen (N)",
            "💎 Distribution of Phosphorus (P)",
            "⚡ Distribution of Potassium (K)",
            "🌡️ Distribution of Temperature",
            "💧 Distribution of Humidity",
            "🧪 Distribution of Soil pH",
            "🌧️ Distribution of Rainfall",
            "💦 Distribution of Moisture",
            "🌬️ Distribution of Wind Speed",
            "🏔️ Distribution of Soil Types",
            "📈 Correlation Heatmap",
            "📊 Boxplot: Nitrogen by Crop",
            "📊 Boxplot: Phosphorus by Crop",
            "📊 Boxplot: Potassium by Crop",
            "📊 Boxplot: Temperature by Crop",
            "📊 Boxplot: Humidity by Crop",
            "📊 Boxplot: Soil pH by Crop",
            "📊 Boxplot: Rainfall by Crop",
            "📊 Boxplot: Moisture by Crop",
            "📊 Boxplot: Wind Speed by Crop"
        ]

        selected_plot = st.selectbox(
            "Choose a mystical visualization to behold:",
            ["Select a plot..."] + plot_options,
            help="Select a plot to display the hidden patterns of the data",
            key="plot_selector"
        )

        if selected_plot != "Select a plot...":
            st.markdown(f"#### {selected_plot}")

            # Generate plot on-the-fly
            try:
                # Load the dataset
                df = pd.read_csv('crop_yield_dataset.csv')

                # Handle missing values
                numerical_cols = ['N', 'P', 'K', 'Temperature', 'Humidity', 'Soil_pH', 'Wind_Speed']
                for col in numerical_cols:
                    if df[col].isnull().sum() > 0:
                        df[col] = df[col].fillna(df[col].mean())

                categorical_cols = ['Soil_Type', 'Crop_Type']
                for col in categorical_cols:
                    if df[col].isnull().sum() > 0:
                        df[col] = df[col].fillna(df[col].mode()[0])

                # Encode for visualization
                le = LabelEncoder()
                df['soil_type_encoded'] = le.fit_transform(df['Soil_Type'])

                # Normalize for visualization
                scaler = StandardScaler()
                df_scaled = df.copy()
                df_scaled[numerical_cols] = scaler.fit_transform(df_scaled[numerical_cols])

                # Create the plot
                fig, ax = plt.subplots(figsize=(10, 6))

                if "Distribution of Nitrogen" in selected_plot:
                    sns.histplot(df_scaled['N'], kde=True, ax=ax)
                    ax.set_title('Distribution of Nitrogen (N)')
                elif "Distribution of Phosphorus" in selected_plot:
                    sns.histplot(df_scaled['P'], kde=True, ax=ax)
                    ax.set_title('Distribution of Phosphorus (P)')
                elif "Distribution of Potassium" in selected_plot:
                    sns.histplot(df_scaled['K'], kde=True, ax=ax)
                    ax.set_title('Distribution of Potassium (K)')
                elif "Distribution of Temperature" in selected_plot:
                    sns.histplot(df_scaled['Temperature'], kde=True, ax=ax)
                    ax.set_title('Distribution of Temperature')
                elif "Distribution of Humidity" in selected_plot:
                    sns.histplot(df_scaled['Humidity'], kde=True, ax=ax)
                    ax.set_title('Distribution of Humidity')
                elif "Distribution of Soil pH" in selected_plot:
                    sns.histplot(df_scaled['Soil_pH'], kde=True, ax=ax)
                    ax.set_title('Distribution of Soil pH')
                elif "Distribution of Rainfall" in selected_plot:
                    # Rainfall not in dataset, show placeholder
                    ax.text(0.5, 0.5, 'Rainfall data not available in current dataset', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title('Distribution of Rainfall (Data Unavailable)')
                elif "Distribution of Moisture" in selected_plot:
                    # Moisture not in dataset, show placeholder
                    ax.text(0.5, 0.5, 'Moisture data not available in current dataset', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title('Distribution of Moisture (Data Unavailable)')
                elif "Distribution of Wind Speed" in selected_plot:
                    sns.histplot(df_scaled['Wind_Speed'], kde=True, ax=ax)
                    ax.set_title('Distribution of Wind Speed')
                elif "Distribution of Soil Types" in selected_plot:
                    sns.histplot(df['soil_type_encoded'], kde=True, ax=ax)
                    ax.set_title('Distribution of Soil Types')
                elif "Correlation Heatmap" in selected_plot:
                    plt.close(fig)  # Close the current figure
                    fig, ax = plt.subplots(figsize=(12, 10))
                    corr_matrix = df_scaled[numerical_cols + ['soil_type_encoded']].corr()
                    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
                    ax.set_title('Correlation Heatmap')
                elif "Boxplot:" in selected_plot:
                    plt.close(fig)  # Close the current figure
                    fig, ax = plt.subplots(figsize=(12, 8))

                    if "Nitrogen by Crop" in selected_plot:
                        sns.boxplot(x='Crop_Type', y='N', data=df_scaled, ax=ax)
                        ax.set_title('Nitrogen by Crop')
                    elif "Phosphorus by Crop" in selected_plot:
                        sns.boxplot(x='Crop_Type', y='P', data=df_scaled, ax=ax)
                        ax.set_title('Phosphorus by Crop')
                    elif "Potassium by Crop" in selected_plot:
                        sns.boxplot(x='Crop_Type', y='K', data=df_scaled, ax=ax)
                        ax.set_title('Potassium by Crop')
                    elif "Temperature by Crop" in selected_plot:
                        sns.boxplot(x='Crop_Type', y='Temperature', data=df_scaled, ax=ax)
                        ax.set_title('Temperature by Crop')
                    elif "Humidity by Crop" in selected_plot:
                        sns.boxplot(x='Crop_Type', y='Humidity', data=df_scaled, ax=ax)
                        ax.set_title('Humidity by Crop')
                    elif "Soil pH by Crop" in selected_plot:
                        sns.boxplot(x='Crop_Type', y='Soil_pH', data=df_scaled, ax=ax)
                        ax.set_title('Soil pH by Crop')
                    elif "Rainfall by Crop" in selected_plot:
                        ax.text(0.5, 0.5, 'Rainfall data not available in current dataset', ha='center', va='center', transform=ax.transAxes)
                        ax.set_title('Rainfall by Crop (Data Unavailable)')
                    elif "Moisture by Crop" in selected_plot:
                        ax.text(0.5, 0.5, 'Moisture data not available in current dataset', ha='center', va='center', transform=ax.transAxes)
                        ax.set_title('Moisture by Crop (Data Unavailable)')
                    elif "Wind Speed by Crop" in selected_plot:
                        sns.boxplot(x='Crop_Type', y='Wind_Speed', data=df_scaled, ax=ax)
                        ax.set_title('Wind Speed by Crop')

                    if "by Crop" in selected_plot and "Rainfall" not in selected_plot and "Moisture" not in selected_plot:
                        ax.tick_params(axis='x', rotation=45)

                # Display the plot
                st.pyplot(fig)
                plt.close(fig)  # Clean up

            except Exception as e:
                st.error(f"🌑 An error occurred while generating the visualization: {str(e)} 🌑")

    except Exception as e:
        st.error(f"🌑 The mystical forces encountered an error: {str(e)} 🌑")

st.markdown("---")
st.markdown('<div class="footer">', unsafe_allow_html=True)
st.markdown("🌟 *This enchanted oracle is a scholarly creation, weaving machine learning magic for crop divination. May your harvests be bountiful!* 🌟")
st.markdown('</div>', unsafe_allow_html=True)
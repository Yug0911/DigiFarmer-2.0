# Crop Recommendation System

A Machine Learning project that recommends the most suitable crop based on soil and environmental conditions.

## Features

- **Data Preprocessing**: Handles missing values, encodes categorical variables, normalizes features
- **Exploratory Data Analysis**: Distribution plots, correlation heatmap, feature comparisons
- **Machine Learning Models**: Random Forest, Decision Tree, SVM, XGBoost
- **Model Evaluation**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- **Web Interface**: Streamlit-based interactive application with magical garden theme
- **API Integrations**: Real-time data fetching from weather, soil, and location APIs
- **Geolocation Support**: Automatic location detection and environmental data retrieval
- **Interactive Visualizations**: On-the-fly EDA plots and data exploration

## Dataset

The project uses a merged dataset (`merged_dataset.csv`) with the following features:
- N (Nitrogen content)
- P (Phosphorus content)
- K (Potassium content)
- Temperature (°C)
- Humidity (%)
- pH
- Rainfall (mm)
- Moisture (%)
- Soil Type (categorical)
- Wind Speed (km/h)
- Crop (target variable)

## Installation

1. Clone or download the project files
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## API Setup

The application integrates with external APIs for real-time environmental data. To enable this functionality:

1. Create a `.env` file in the project root directory
2. Add your API keys:
   ```
   OPENWEATHER_API_KEY=your_openweather_api_key_here
   ```
3. Get your OpenWeatherMap API key from [openweathermap.org](https://openweathermap.org/api)
4. Test your API key setup by running:
   ```bash
   python test_api.py
   ```

## Usage

### Training the Model

Run the preprocessing script to train models and generate visualizations:
```bash
python preprocessing.py
```

This will:
- Load and preprocess the data
- Generate EDA plots (saved as PNG files)
- Train multiple ML models
- Evaluate and compare model performance
- Save the best model and preprocessors

### Running the Web Application

Launch the Streamlit web app:
```bash
streamlit run app.py
```

The app will open in your default web browser with a magical garden theme. You have two options:

#### Manual Input
Enter the environmental parameters manually and click "Recommend Crop" to get predictions.

#### Automatic Data Fetching (Recommended)
1. Check the "Use my current location for automatic data fetching" option
2. Click "Get My Location & Fetch Data" to automatically retrieve:
   - Location coordinates and city information
   - Real-time weather data (temperature, humidity, wind speed, rainfall)
   - Soil data (pH, organic carbon, moisture) from regional databases
3. The parameters will be auto-filled with current environmental conditions
4. Click "Recommend Crop" to get AI-powered crop recommendations

The app also includes interactive data visualizations for exploring the dataset patterns.

## Project Structure

```
├── merged_dataset.csv          # Dataset file
├── crop_yield_dataset.csv      # Additional dataset for visualizations
├── preprocessing.py            # Data preprocessing and model training
├── app.py                      # Streamlit web application with API integrations
├── api_utils.py                # API utilities for weather, soil, and location data
├── test_api.py                 # API key testing script
├── fill_nulls.py               # Data cleaning utilities
├── preprocessing.ipynb         # Jupyter notebook for data exploration
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (API keys)
├── README.md                   # This file
├── scaler.pkl                  # Saved StandardScaler
├── label_encoder.pkl           # Saved LabelEncoder for soil types
├── target_encoder.pkl          # Saved LabelEncoder for crops
├── best_crop_model.pkl         # Saved best ML model
└── *.png                       # EDA visualization plots
```

## Model Performance

The Random Forest model achieved the best performance:
- **Accuracy**: 75.80%
- **Precision**: 75.76%
- **Recall**: 75.80%
- **F1-Score**: 75.75%

## Academic Project

This project was developed for academic submission, demonstrating:
- Data preprocessing techniques
- Exploratory data analysis
- Machine learning model implementation
- Model evaluation and selection
- Web application development with Streamlit
- API integration and real-time data fetching
- Geolocation services and environmental data retrieval
- Interactive data visualization
- Full-stack ML application development

## Note

This is a demonstration project. For real-world agricultural recommendations, consult local agricultural experts and consider additional factors not included in this model.
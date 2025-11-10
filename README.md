# Crop Recommendation System

A Machine Learning project that recommends the most suitable crop based on soil and environmental conditions.

## Features

- **Data Preprocessing**: Handles missing values, encodes categorical variables, normalizes features
- **Exploratory Data Analysis**: Distribution plots, correlation heatmap, feature comparisons
- **Machine Learning Models**: Random Forest, Decision Tree, SVM, XGBoost
- **Model Evaluation**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- **Web Interface**: Streamlit-based interactive application

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

The app will open in your default web browser. Enter the environmental parameters and click "Recommend Crop" to get predictions.

## Project Structure

```
├── merged_dataset.csv      # Dataset file
├── preprocessing.py        # Data preprocessing and model training
├── app.py                  # Streamlit web application
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── scaler.pkl             # Saved StandardScaler
├── label_encoder.pkl      # Saved LabelEncoder for soil types
├── target_encoder.pkl     # Saved LabelEncoder for crops
├── best_crop_model.pkl    # Saved best ML model
└── *.png                  # EDA visualization plots
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

## Note

This is a demonstration project. For real-world agricultural recommendations, consult local agricultural experts and consider additional factors not included in this model.
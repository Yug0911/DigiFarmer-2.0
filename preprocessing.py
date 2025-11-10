import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
from scipy.stats import randint, uniform

# Load the dataset
df = pd.read_csv('crop_yield_dataset.csv')

# Display first few rows
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Handle missing values (if any)
# For numerical columns, fill with mean
numerical_cols = ['N', 'P', 'K', 'Temperature', 'Humidity', 'Soil_pH', 'Wind_Speed']
for col in numerical_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mean())

# For categorical columns, fill with mode
categorical_cols = ['Soil_Type', 'Crop_Type']
for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mode()[0])

# Encode Soil Type
le = LabelEncoder()
df['soil_type_encoded'] = le.fit_transform(df['Soil_Type'])

# Encode target variable
le_target = LabelEncoder()
df['crop_encoded'] = le_target.fit_transform(df['Crop_Type'])

# Normalize numerical features
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Feature Engineering - Add interaction features and polynomial features
df['n_p_ratio'] = df['N'] / (df['P'] + 1)  # Avoid division by zero
df['n_k_ratio'] = df['N'] / (df['K'] + 1)
df['p_k_ratio'] = df['P'] / (df['K'] + 1)
df['temp_humidity'] = df['Temperature'] * df['Humidity']
df['wind_temp'] = df['Wind_Speed'] * df['Temperature']

# Add polynomial features for key nutrients
df['n_squared'] = df['N'] ** 2
df['p_squared'] = df['P'] ** 2
df['k_squared'] = df['K'] ** 2
df['ph_squared'] = df['Soil_pH'] ** 2

# Environmental interactions
df['temp_ph'] = df['Temperature'] * df['Soil_pH']
df['humidity_ph'] = df['Humidity'] * df['Soil_pH']

# Features and target
X = df[numerical_cols + ['soil_type_encoded', 'n_p_ratio', 'n_k_ratio', 'p_k_ratio', 'temp_humidity', 'wind_temp']]
y = df['crop_encoded']

# Train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Save preprocessor
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le, 'label_encoder.pkl')
joblib.dump(le_target, 'target_encoder.pkl')

# EDA
# Distribution plots
for col in numerical_cols + ['soil_type_encoded']:
    plt.figure(figsize=(8, 6))
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.savefig(f'distribution_{col}.png')
    plt.close()

# Correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(df[numerical_cols + ['soil_type_encoded']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.savefig('correlation_heatmap.png')
plt.close()

# Comparison of features by crop
for col in numerical_cols:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Crop_Type', y=col, data=df)
    plt.title(f'{col} by Crop')
    plt.xticks(rotation=45)
    plt.savefig(f'boxplot_{col}_by_crop.png')
    plt.close()

# Hyperparameter Tuning
print("Starting Hyperparameter Tuning...")

# Random Forest Tuning
rf_param_dist = {
    'n_estimators': randint(100, 500),
    'max_depth': [10, 15, 20, 25, None],
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 5),
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False]
}

rf_random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    rf_param_dist,
    n_iter=50,
    cv=3,
    scoring='f1_weighted',
    n_jobs=-1,
    random_state=42
)

rf_random_search.fit(X_train, y_train)
best_rf = rf_random_search.best_estimator_
print(f"Best Random Forest Params: {rf_random_search.best_params_}")
print(f"Best RF CV Score: {rf_random_search.best_score_:.4f}")

# XGBoost Tuning
xgb_param_dist = {
    'n_estimators': randint(100, 500),
    'max_depth': randint(3, 15),
    'learning_rate': uniform(0.01, 0.3),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4),
    'gamma': uniform(0, 0.5),
    'min_child_weight': randint(1, 10)
}

xgb_random_search = RandomizedSearchCV(
    XGBClassifier(random_state=42),
    xgb_param_dist,
    n_iter=50,
    cv=3,
    scoring='f1_weighted',
    n_jobs=-1,
    random_state=42
)

xgb_random_search.fit(X_train, y_train)
best_xgb = xgb_random_search.best_estimator_
print(f"Best XGBoost Params: {xgb_random_search.best_params_}")
print(f"Best XGB CV Score: {xgb_random_search.best_score_:.4f}")

# Decision Tree Tuning
dt_param_grid = {
    'max_depth': [10, 15, 20, 25, None],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4, 6],
    'criterion': ['gini', 'entropy'],
    'max_features': ['sqrt', 'log2', None]
}

dt_grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    dt_param_grid,
    cv=3,
    scoring='f1_weighted',
    n_jobs=-1
)

dt_grid_search.fit(X_train, y_train)
best_dt = dt_grid_search.best_estimator_
print(f"Best Decision Tree Params: {dt_grid_search.best_params_}")
print(f"Best DT CV Score: {dt_grid_search.best_score_:.4f}")

# Build and Train Models with Tuned Parameters
from sklearn.model_selection import cross_val_score

models = {
    'Random Forest': best_rf,
    'XGBoost': best_xgb,
    'Decision Tree': best_dt
}

# Cross-validation for better evaluation
for name, model in models.items():
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_weighted')
    print(f'{name} CV F1-Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})')

trained_models = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    trained_models[name] = model
    print(f'{name} trained.')

# Evaluate Models
results = {}
for name, model in trained_models.items():
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)

    results[name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Confusion Matrix': cm
    }

    print(f'{name} Results:')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-Score: {f1:.4f}')
    print(f'Confusion Matrix:\n{cm}')
    print('-' * 50)

# Select Best Model
best_model_name = max(results, key=lambda x: results[x]['F1-Score'])
best_model = trained_models[best_model_name]
print(f'Best Model: {best_model_name} with F1-Score: {results[best_model_name]["F1-Score"]:.4f}')

# Save best model
joblib.dump(best_model, 'best_crop_model.pkl')

# Prediction Function
def predict_crop(n, p, k, temperature, humidity, ph, rainfall, moisture, soil_type, windspeed):
    # Load preprocessors
    scaler = joblib.load('scaler.pkl')
    le = joblib.load('label_encoder.pkl')
    le_target = joblib.load('target_encoder.pkl')
    model = joblib.load('best_crop_model.pkl')

    # Prepare input with all features in correct order
    input_data = pd.DataFrame({
        'N': [n], 'P': [p], 'K': [k], 'Temperature': [temperature],
        'Humidity': [humidity], 'Soil_pH': [ph], 'Wind_Speed': [windspeed]
    })

    # Add engineered features
    input_data['n_p_ratio'] = input_data['N'] / (input_data['P'] + 1)
    input_data['n_k_ratio'] = input_data['N'] / (input_data['K'] + 1)
    input_data['p_k_ratio'] = input_data['P'] / (input_data['K'] + 1)
    input_data['temp_humidity'] = input_data['Temperature'] * input_data['Humidity']
    input_data['wind_temp'] = input_data['Wind_Speed'] * input_data['Temperature']
    input_data['n_squared'] = input_data['N'] ** 2
    input_data['p_squared'] = input_data['P'] ** 2
    input_data['k_squared'] = input_data['K'] ** 2
    input_data['ph_squared'] = input_data['Soil_pH'] ** 2
    input_data['temp_ph'] = input_data['Temperature'] * input_data['Soil_pH']
    input_data['humidity_ph'] = input_data['Humidity'] * input_data['Soil_pH']

    # Encode soil type
    soil_encoded = le.transform([soil_type])[0]
    input_data['soil_type_encoded'] = soil_encoded

    # Ensure correct column order
    feature_cols = numerical_cols + ['soil_type_encoded', 'n_p_ratio', 'n_k_ratio', 'p_k_ratio', 'temp_humidity', 'wind_temp']
    input_data = input_data[feature_cols]

    # Scale numerical features (excluding engineered features)
    input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])

    # Predict
    prediction_encoded = model.predict(input_data)

    # Decode prediction
    prediction = le_target.inverse_transform(prediction_encoded)

    return prediction[0]

# Test the function
print(predict_crop(90, 42, 43, 20.87, 82.00, 6.50, 202.93, 29.44, 'Loamy', 10.10))
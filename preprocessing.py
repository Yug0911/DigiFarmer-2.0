import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC  # (kept import in case you want to try it)
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from scipy.stats import randint, uniform
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ---------------------------------------
# Configuration
# ---------------------------------------
DATA_PATH = 'crop_yield_dataset.csv'  # adjust if your CSV lives elsewhere

# Turn off EDA image generation by default to save time
DO_EDA = False  # set True for one-time EDA image export

ARTIFACTS = ['scaler.pkl', 'label_encoder.pkl', 'target_encoder.pkl', 'best_crop_model.pkl']

# Features used for scaling
numerical_cols = ['N', 'P', 'K', 'Temperature', 'Humidity', 'Soil_pH', 'Wind_Speed']

# Engineered features used in training/prediction
engineered_cols = ['soil_type_encoded',
                   'n_p_ratio', 'n_k_ratio', 'p_k_ratio',
                   'temp_humidity', 'wind_temp']

# Final feature order for the model
feature_cols = numerical_cols + engineered_cols


# ---------------------------------------
# Helpers
# ---------------------------------------
def _artifacts_exist() -> bool:
    """Check if all artifacts exist in the current working directory."""
    return all(os.path.exists(f) for f in ARTIFACTS)


def _save_preprocessors(scaler: StandardScaler, le: LabelEncoder, le_target: LabelEncoder):
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(le, 'label_encoder.pkl')
    joblib.dump(le_target, 'target_encoder.pkl')


def _load_preprocessors_model():
    scaler = joblib.load('scaler.pkl')
    le = joblib.load('label_encoder.pkl')
    le_target = joblib.load('target_encoder.pkl')
    model = joblib.load('best_crop_model.pkl')
    return scaler, le, le_target, model


def _do_eda(df: pd.DataFrame, le: LabelEncoder):
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

    # Boxplots by crop
    for col in numerical_cols:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Crop_Type', y=col, data=df)
        plt.title(f'{col} by Crop')
        plt.xticks(rotation=45)
        plt.savefig(f'boxplot_{col}_by_crop.png')
        plt.close()


def _prepare_dataframe(df: pd.DataFrame):
    """Handle missing values, encode labels, scale numerics, and engineer features.
       Returns processed df, scaler, label encoders, and y (encoded target)."""

    # Handle missing values
    for col in numerical_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mean())

    for col in ['Soil_Type', 'Crop_Type']:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])

    # Encode Soil Type
    le = LabelEncoder()
    df['soil_type_encoded'] = le.fit_transform(df['Soil_Type'])

    # Encode target variable
    le_target = LabelEncoder()
    df['crop_encoded'] = le_target.fit_transform(df['Crop_Type'])

    # Scale numerical features
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    # Feature Engineering
    df['n_p_ratio']      = df['N'] / (df['P'] + 1)   # avoid division by zero
    df['n_k_ratio']      = df['N'] / (df['K'] + 1)
    df['p_k_ratio']      = df['P'] / (df['K'] + 1)
    df['temp_humidity']  = df['Temperature'] * df['Humidity']
    df['wind_temp']      = df['Wind_Speed'] * df['Temperature']

    # (You computed more polynomial features, but they were not used in X in the original code.
    #  Keeping behavior consistent: they’re not included in feature_cols.)

    X = df[feature_cols].copy()
    y = df['crop_encoded'].copy()
    return df, X, y, scaler, le, le_target


def train_and_persist():
    """Train once (if needed) and persist artifacts."""
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    print(df.head())
    print(df.isnull().sum())

    print("Preparing data...")
    df, X, y, scaler, le, le_target = _prepare_dataframe(df)

    if DO_EDA:
        print("Running EDA and saving charts...")
        _do_eda(df, le)

    print("Train-test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Save preprocessors early (reusable even if training fails later)
    _save_preprocessors(scaler, le, le_target)

    print("Starting Hyperparameter Tuning...")

    # Random Forest
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

    # XGBoost
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
        XGBClassifier(random_state=42, n_jobs=-1, tree_method="hist"),
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

    # Decision Tree
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

    # Cross-validation for the tuned models
    models = {
        'Random Forest': best_rf,
        'XGBoost': best_xgb,
        'Decision Tree': best_dt
    }
    for name, model in models.items():
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_weighted', n_jobs=-1)
        print(f'{name} CV F1-Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})')

    # Fit tuned models and evaluate
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
        print(f'{name} trained.')

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

    # Select and save best model
    best_model_name = max(results, key=lambda x: results[x]['F1-Score'])
    best_model = trained_models[best_model_name]
    joblib.dump(best_model, 'best_crop_model.pkl')
    print(f'Best Model: {best_model_name} with F1-Score: {results[best_model_name]["F1-Score"]:.4f}')
    print('Artifacts saved: scaler.pkl, label_encoder.pkl, target_encoder.pkl, best_crop_model.pkl')


def ensure_trained_once():
    """Run training only if artifacts are missing."""
    if _artifacts_exist():
        print("Artifacts already exist; skipping training.")
    else:
        print("Artifacts not found; training now (one-time)...")
        train_and_persist()


# ---------------------------------------
# Public Prediction API
# ---------------------------------------
def predict_crop(n, p, k, temperature, humidity, ph, rainfall, moisture, soil_type, windspeed):
    """
    Predict the crop using persisted artifacts.
    rainfall and moisture are accepted to keep signature identical to your app,
    but they are not used in the current feature set (consistent with original code).
    """
    if not _artifacts_exist():
        # Safety: train once if user tries to predict before first training
        ensure_trained_once()

    scaler, le, le_target, model = _load_preprocessors_model()

    input_data = pd.DataFrame({
        'N': [n],
        'P': [p],
        'K': [k],
        'Temperature': [temperature],
        'Humidity': [humidity],
        'Soil_pH': [ph],
        'Wind_Speed': [windspeed],
    })

    # Engineer features (mirror training)
    input_data['n_p_ratio']     = input_data['N'] / (input_data['P'] + 1)
    input_data['n_k_ratio']     = input_data['N'] / (input_data['K'] + 1)
    input_data['p_k_ratio']     = input_data['P'] / (input_data['K'] + 1)
    input_data['temp_humidity'] = input_data['Temperature'] * input_data['Humidity']
    input_data['wind_temp']     = input_data['Wind_Speed'] * input_data['Temperature']

    # Encode soil type
    soil_encoded = le.transform([soil_type])[0]
    input_data['soil_type_encoded'] = soil_encoded

    # Ensure correct column order
    input_data = input_data.reindex(columns=feature_cols)

    # Scale numerical features
    input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])

    # Predict
    prediction_encoded = model.predict(input_data)
    prediction = le_target.inverse_transform(prediction_encoded)
    return prediction[0]


# ---------------------------------------
# One-time training guard on import
# ---------------------------------------
# This makes sure that importing this module in Streamlit/Flask won't retrain every time:
ensure_trained_once()

# Optional: quick manual test when run directly
if __name__ == '__main__':
    print(predict_crop(90, 42, 43, 20.87, 82.00, 6.50, 202.93, 29.44, 'Loamy', 10.10))

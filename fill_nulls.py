import pandas as pd
import numpy as np

# Load both datasets
merged_df = pd.read_csv('merged_dataset.csv')
yield_df = pd.read_csv('crop_yield_dataset.csv')

print("Merged dataset shape:", merged_df.shape)
print("Yield dataset shape:", yield_df.shape)

# Check null values in merged dataset
print("\nNull values in merged dataset:")
print(merged_df.isnull().sum())

# Check unique crops in both datasets
print("\nUnique crops in merged dataset:", merged_df['crop'].unique())
print("Unique crops in yield dataset:", yield_df['Crop_Type'].unique())

# Create mapping for soil types if needed
soil_mapping = {
    'Loamy': 'Loamy',
    'Sandy': 'Sandy',
    'Clayey': 'Clayey',
    'Black': 'Black',
    'Red': 'Red',
    'Peaty': 'Peaty'
}

# Create a mapping for crop names to match yield dataset
crop_mapping = {
    'rice': 'Rice',
    'wheat': 'Wheat',
    'corn': 'Corn',
    'maize': 'Corn',
    'barley': 'Barley',
    'soybean': 'Soybean',
    'cotton': 'Cotton',
    'sugarcane': 'Sugarcane',
    'tomato': 'Tomato',
    'potato': 'Potato',
    'sunflower': 'Sunflower'
}

# Create a mapping for soil types
soil_mapping = {
    'Loamy': 'Loamy',
    'Sandy': 'Sandy',
    'Clayey': 'Clayey',
    'Black': 'Black',
    'Red': 'Red',
    'Peaty': 'Peaty',
    '2': 'Loamy',  # Assuming numeric codes map to types
    '3': 'Sandy',
    '4': 'Clayey',
    '5': 'Black'
}

# Fill date and soil_quality based on crop and soil_type matching
for idx, row in merged_df.iterrows():
    if pd.isna(row['date']) or pd.isna(row['soil_quality']):
        crop = row['crop']
        soil_type = str(row['soil_type'])

        # Map crop name
        mapped_crop = crop_mapping.get(crop.lower(), crop.title())

        # Map soil type
        mapped_soil = soil_mapping.get(soil_type, soil_type)

        # Find matching rows in yield dataset
        matching_rows = yield_df[
            (yield_df['Crop_Type'] == mapped_crop) &
            (yield_df['Soil_Type'] == mapped_soil)
        ]

        if not matching_rows.empty:
            # Use random sample from matching rows for variety
            sample_row = matching_rows.sample(n=1).iloc[0]

            # Fill date
            if pd.isna(row['date']):
                merged_df.at[idx, 'date'] = sample_row['Date']

            # Fill soil_quality
            if pd.isna(row['soil_quality']):
                merged_df.at[idx, 'soil_quality'] = sample_row['Soil_Quality']
        else:
            # If no exact match, try just crop match
            crop_only_matches = yield_df[yield_df['Crop_Type'] == mapped_crop]
            if not crop_only_matches.empty:
                sample_row = crop_only_matches.sample(n=1).iloc[0]

                if pd.isna(row['date']):
                    merged_df.at[idx, 'date'] = sample_row['Date']
                if pd.isna(row['soil_quality']):
                    merged_df.at[idx, 'soil_quality'] = sample_row['Soil_Quality']

print("\nAfter filling nulls:")
print(merged_df.isnull().sum())

# Save the updated dataset
merged_df.to_csv('merged_dataset_filled.csv', index=False)
print("\nUpdated dataset saved as 'merged_dataset_filled.csv'")

# Show sample of filled data
print("\nSample of filled data:")
print(merged_df[['crop', 'soil_type', 'date', 'soil_quality']].head(10))
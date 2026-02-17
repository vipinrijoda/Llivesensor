# quick_fix.py
import pandas as pd
import numpy as np
from datetime import datetime
from sensor.configuration.mongo_db_connection import MongoDBClient
from sensor.ml.model.estimator import ModelResolver
from sensor.utils.main_utils import load_object
from sensor.constant.training_pipeline import SAVED_MODEL_DIR
import os

print("="*50)
print("QUICK FIX - Generating Predictions")
print("="*50)

# Columns to drop from your schema
COLUMNS_TO_DROP = ['br_000', 'bq_000', 'bp_000', 'ab_000', 'cr_000', 'bo_000', 'bn_000']

# Step 1: Check if model exists
resolver = ModelResolver(model_dir=SAVED_MODEL_DIR)
if not resolver.is_model_exists():
    print("ERROR: No model found! Please run training first.")
    exit()

model_path = resolver.get_best_model_path()
print("Model found at: " + model_path)

# Step 2: Load model
model = load_object(model_path)
print("Model loaded successfully")

# Step 3: Find your CSV file
csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
print("Found CSV files: " + str(csv_files))

if not csv_files:
    print("ERROR: No CSV files found!")
    exit()

# Use first CSV file
csv_file = csv_files[0]
print("Using file: " + csv_file)

# Step 4: Load data with proper handling of 'na' values
print("Loading data and handling 'na' values...")
df = pd.read_csv(csv_file, na_values=['na', 'NaN', 'NA', 'null', ''])
print("Loaded " + str(len(df)) + " rows from " + csv_file)

# Take 100 samples
sample_df = df.head(100).copy()
print("Taking first 100 rows for predictions")

# Step 5: Drop columns that were dropped during training
print("Dropping columns that were not used in training...")
for col in COLUMNS_TO_DROP:
    if col in sample_df.columns:
        sample_df = sample_df.drop(col, axis=1)
        print("  Dropped: " + col)

# Remove 'class' column if exists (target variable)
if 'class' in sample_df.columns:
    sample_df = sample_df.drop('class', axis=1)
    print("Dropped 'class' column")

# Step 6: Convert all columns to numeric, coercing errors to NaN
print("Converting data to numeric...")
for col in sample_df.columns:
    sample_df[col] = pd.to_numeric(sample_df[col], errors='coerce')

# Check for NaN values
nan_count = sample_df.isna().sum().sum()
if nan_count > 0:
    print("Found " + str(nan_count) + " NaN values. Filling with 0...")
    sample_df = sample_df.fillna(0)

# Step 7: Get the features the model expects
expected_features = None
if hasattr(model, 'preprocessor'):
    if hasattr(model.preprocessor, 'feature_names_in_'):
        expected_features = model.preprocessor.feature_names_in_.tolist()
        print("Model expects " + str(len(expected_features)) + " features")
        print("First few expected features: " + str(expected_features[:5]))

if expected_features:
    # Keep only the columns that the model expects
    available_features = [col for col in expected_features if col in sample_df.columns]
    missing_features = [col for col in expected_features if col not in sample_df.columns]
    
    print("Available features in data: " + str(len(available_features)))
    print("Missing features (will be filled with 0): " + str(len(missing_features)))
    
    # Add missing features with 0
    for col in missing_features:
        sample_df[col] = 0
    
    # Reorder columns to match model's expected order
    sample_df = sample_df[expected_features]
    print("Data now has " + str(sample_df.shape[1]) + " columns in correct order")
    
    # Final check for any remaining non-numeric data
    print("Final data types check...")
    dtypes = sample_df.dtypes.unique()
    print("Data types: " + str(dtypes))

# Step 8: Make predictions
try:
    predictions = model.predict(sample_df)
    print("Made " + str(len(predictions)) + " predictions")
    
    # Get probabilities if available
    probabilities = None
    if hasattr(model, 'predict_proba'):
        try:
            proba = model.predict_proba(sample_df)
            probabilities = [max(p) for p in proba]
            print("Got probability scores")
        except Exception as e:
            print("Could not get probabilities: " + str(e))
            probabilities = [0.95] * len(predictions)
    else:
        probabilities = [0.95] * len(predictions)
        
    # Count predictions
    fault_count = sum(predictions)
    normal_count = len(predictions) - fault_count
    print("Predictions summary: " + str(int(fault_count)) + " faults, " + str(int(normal_count)) + " normal")
        
except Exception as e:
    print("Error during prediction: " + str(e))
    import traceback
    traceback.print_exc()
    exit()

# Step 9: Save to MongoDB
try:
    client = MongoDBClient()
    db = client.database
    collection = db['predictions']

    # Clear existing predictions
    collection.delete_many({})
    print("Cleared old predictions")

    # Create new records
    records = []
    for i in range(len(sample_df)):
        record = {
            'timestamp': datetime.now(),
            'prediction': int(predictions[i]),
            'confidence': float(probabilities[i]),
            'reviewed': False
        }
        
        # Add first 10 sensor values for display
        for j, col in enumerate(sample_df.columns[:10]):
            # Ensure value is float
            val = sample_df.iloc[i][col]
            if pd.isna(val):
                val = 0.0
            record[col] = float(val)
        
        records.append(record)

    # Insert to MongoDB
    result = collection.insert_many(records)
    print("Saved " + str(len(records)) + " predictions to 'predictions' collection")

    # Step 10: Verify
    count = collection.count_documents({})
    print("Collection 'predictions' now has " + str(count) + " documents")

    # Step 11: Show sample
    print("\nSample prediction record:")
    sample = collection.find_one()
    if sample:
        print("  Prediction: " + str(sample.get('prediction')))
        print("  Confidence: " + str(sample.get('confidence')))
        print("  Timestamp: " + str(sample.get('timestamp')))
        print("  First sensor value: " + str(list(sample.items())[3]))

except Exception as e:
    print("Error saving to MongoDB: " + str(e))
    import traceback
    traceback.print_exc()

print("="*50)
print("DONE! Refresh your dashboard now!")
print("="*50)
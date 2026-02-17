from sensor.exception import SensorException
from sensor.logger import logging
import os
import sys
from sensor.utils2 import dump_csv_file_to_mongodb_collection
from sensor.configuration.mongo_db_connection import MongoDBClient
from sensor.ml.model.estimator import ModelResolver, TargetValueMapping
from sensor.pipeline import training_pipeline
from sensor.pipeline.training_pipeline import TrainPipeline
from sensor.utils.main_utils import read_yaml_file, load_object
from sensor.constant.training_pipeline import SAVED_MODEL_DIR

from fastapi import FastAPI
from sensor.constant.application import APP_HOST, APP_PORT
from starlette.responses import RedirectResponse
from uvicorn import run as app_run
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile
import pandas as pd
import numpy as np

app = FastAPI()

origins = ["*"]
# Cross-Origin Resource Sharing (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")

@app.get("/train")
async def train():
    try:   
        training_pipeline = TrainPipeline()
        if training_pipeline.is_pipeline_running:
            return Response("Training pipeline is already running.")
        
        training_pipeline.run_pipeline()
        return Response("Training successfully completed!")
    except Exception as e:
        return Response(f"Error Occured! {e}")

@app.get("/predict")
async def predict():
    try:
        # First, let's check what features the model expects
        model_resolver = ModelResolver(model_dir=SAVED_MODEL_DIR)
        
        if not model_resolver.is_model_exists():
            return Response("Model is not available. Please train the model first.")
        
        best_model_path = model_resolver.get_best_model_path()
        model = load_object(file_path=best_model_path)
        
        # Get the feature names that the model expects
        expected_features = model.preprocessor.feature_names_in_.tolist()
        print(f"Model expects {len(expected_features)} features: {expected_features[:10]}...")  # Print first 10
        
        # Create sample data with ALL expected features
        # Initialize with zeros for all features
        sample_dict = {}
        
        # Generate sample values for each expected feature
        for i, feature in enumerate(expected_features):
            # Create 5 sample rows with different values
            sample_dict[feature] = [
                0.1 + (i % 10) * 0.1,
                0.2 + (i % 10) * 0.1,
                0.3 + (i % 10) * 0.1,
                0.4 + (i % 10) * 0.1,
                0.5 + (i % 10) * 0.1
            ]
        
        # Create dataframe from sample data
        df = pd.DataFrame(sample_dict)
        
        print(f"Created dataframe with {df.shape[1]} features and {df.shape[0]} rows")
        
        # Make predictions
        predictions = model.predict(df)
        print(f"Predictions: {predictions}")
        
        # Convert predictions to readable format
        target_mapping = TargetValueMapping()
        reverse_map = target_mapping.reverse_mapping()
        
        # Add predictions to dataframe
        df['prediction'] = predictions
        df['fault_status'] = df['prediction'].map(reverse_map)
        
        # Prepare response
        result = {
            "status": "success",
            "message": "Predictions generated successfully",
            "total_samples": len(df),
            "faults_detected": int(sum(predictions)),
            "no_fault": int(len(predictions) - sum(predictions)),
            "feature_count": len(expected_features),
            "sample_features_used": expected_features[:5],  # Show first 5 features
            "predictions": df[['prediction', 'fault_status']].to_dict(orient='records')
        }
        
        return result

    except AttributeError as e:
        print(f"Attribute error: {str(e)}")
        return Response(f"Model attribute error: {str(e)}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise SensorException(e, sys)


@app.post("/predict/file")
async def predict_file(file: UploadFile = File(...)):
    try:
        # Read uploaded CSV file
        df = pd.read_csv(file.file)
        
        print(f"Uploaded file has {df.shape[1]} features: {df.columns[:10].tolist()}...")
        
        # Remove 'class' column if it exists
        if 'class' in df.columns:
            df = df.drop('class', axis=1)
            print("Removed 'class' column")
        
        # Handle missing values
        df = df.replace('na', np.nan)
        df = df.replace('NaN', np.nan)
        df = df.replace('', np.nan)
        
        # Convert all columns to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Fill NaN values with 0
        df = df.fillna(0)
        
        print(f"After preprocessing: {df.shape[1]} features, {df.shape[0]} rows")
        
        model_resolver = ModelResolver(model_dir=SAVED_MODEL_DIR)
        
        if not model_resolver.is_model_exists():
            return {"status": "error", "message": "Model is not available"}
        
        best_model_path = model_resolver.get_best_model_path()
        model = load_object(file_path=best_model_path)
        
        # Get expected features
        expected_features = model.preprocessor.feature_names_in_.tolist()
        
        # Check features
        uploaded_features = df.columns.tolist()
        common_features = set(uploaded_features) & set(expected_features)
        missing_features = set(expected_features) - set(uploaded_features)
        
        # Add missing features
        for feature in missing_features:
            df[feature] = 0
        
        # Select features in correct order
        df = df[expected_features]
        
        # Make predictions
        predictions = model.predict(df)
        
        # Convert to readable format
        target_mapping = TargetValueMapping()
        reverse_map = target_mapping.reverse_mapping()
        
        # Prepare results (limit to first 100 to avoid huge response)
        max_samples = min(100, len(df))
        results = []
        for i in range(max_samples):
            results.append({
                "sample_id": i+1,
                "prediction": int(predictions[i]),
                "fault_status": reverse_map[int(predictions[i])]
            })
        
        # Return proper JSON
        return {
            "status": "success",
            "total_samples": len(df),
            "faults_detected": int(sum(predictions)),
            "feature_stats": {
                "expected_features": len(expected_features),
                "features_provided": len(uploaded_features),
                "matching_features": len(common_features),
                "missing_features_filled": len(missing_features)
            },
            "results": results,
            "note": "Showing first 100 results only"
        }

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "APS Sensor Fault Detection API is running"}

def main():
    try:
        training_pipeline = TrainPipeline()
        training_pipeline.run_pipeline()
    except Exception as e:
        print(e)
        logging.exception(e)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=APP_HOST, port=APP_PORT)




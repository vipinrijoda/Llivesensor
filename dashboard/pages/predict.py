import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys
import os
import io
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent.parent))

from sensor.ml.model.estimator import ModelResolver
from sensor.utils.main_utils import load_object
from sensor.constant.training_pipeline import SAVED_MODEL_DIR
from sensor.configuration.mongo_db_connection import MongoDBClient

# Columns to drop from your schema
COLUMNS_TO_DROP = ['br_000', 'bq_000', 'bp_000', 'ab_000', 'cr_000', 'bo_000', 'bn_000']

def show():
    st.title("🔮 Predict on New Data")
    
    st.markdown("""
    Upload a CSV file with sensor readings to get predictions from your trained model.
    The file should contain the same sensor columns your model was trained on.
    """)
    
    # Check if model exists
    model_dir = os.path.join(Path(__file__).parent.parent.parent, "saved_models")
    
    if not os.path.exists(model_dir):
        st.error("❌ No saved_models directory found. Please train a model first.")
        return
    
    model_versions = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
    
    if not model_versions:
        st.error("❌ No trained models found. Please run training pipeline first.")
        return
    
    # Model selection
    selected_version = st.selectbox("Select Model Version", model_versions)
    model_path = os.path.join(model_dir, selected_version, "model.pkl")
    
    # Load model
    with st.spinner("Loading model..."):
        try:
            model = load_object(model_path)
            st.success(f"✅ Model loaded successfully")
            
            # Show model info
            col1, col2 = st.columns(2)
            with col1:
                if hasattr(model, 'model'):
                    st.info(f"**Model Type:** {type(model.model).__name__}")
            with col2:
                st.info(f"**Version:** {selected_version}")
            
            # Get expected features
            expected_features = None
            if hasattr(model, 'preprocessor'):
                if hasattr(model.preprocessor, 'feature_names_in_'):
                    expected_features = model.preprocessor.feature_names_in_.tolist()
                    st.info(f"**Expected Features:** {len(expected_features)} sensors")
                    
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            return
    
    st.markdown("---")
    
    # File upload section
    st.subheader("📤 Upload CSV File")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file", 
        type=['csv'],
        help="Upload a CSV file with sensor readings"
    )
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ File loaded: {uploaded_file.name}")
            st.info(f"**Shape:** {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Show sample of uploaded data
            with st.expander("📋 Preview Uploaded Data"):
                st.dataframe(df.head(10))
            
            # Data preprocessing section
            st.subheader("🔄 Data Preprocessing")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Before Preprocessing:**")
                st.write(f"- Rows: {df.shape[0]}")
                st.write(f"- Columns: {df.shape[1]}")
                st.write(f"- Missing values: {df.isna().sum().sum()}")
            
            # Make a copy for processing
            processed_df = df.copy()
            
            # Drop columns that were dropped during training
            dropped = []
            for col in COLUMNS_TO_DROP:
                if col in processed_df.columns:
                    processed_df = processed_df.drop(col, axis=1)
                    dropped.append(col)
            
            if dropped:
                st.write(f"✅ Dropped columns: {', '.join(dropped)}")
            
            # Remove 'class' column if exists
            if 'class' in processed_df.columns:
                processed_df = processed_df.drop('class', axis=1)
                st.write("✅ Dropped 'class' column")
            
            # Convert to numeric
            for col in processed_df.columns:
                processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
            
            # Fill NaN values
            nan_count = processed_df.isna().sum().sum()
            if nan_count > 0:
                processed_df = processed_df.fillna(0)
                st.write(f"✅ Filled {nan_count} missing values with 0")
            
            with col2:
                st.write("**After Preprocessing:**")
                st.write(f"- Rows: {processed_df.shape[0]}")
                st.write(f"- Columns: {processed_df.shape[1]}")
                st.write(f"- Missing values: {processed_df.isna().sum().sum()}")
            
            # Check if features match
            if expected_features:
                # Check which expected features are in the data
                available = [col for col in expected_features if col in processed_df.columns]
                missing = [col for col in expected_features if col not in processed_df.columns]
                
                if missing:
                    st.warning(f"⚠️ Missing {len(missing)} expected features. They will be filled with 0.")
                    with st.expander("View missing features"):
                        st.write(missing[:20])
                        if len(missing) > 20:
                            st.write(f"... and {len(missing)-20} more")
                    
                    # Add missing columns with 0
                    for col in missing:
                        processed_df[col] = 0
                
                # Reorder columns to match model's expected order
                processed_df = processed_df[expected_features]
                st.success(f"✅ Data aligned: {processed_df.shape[1]} features ready for prediction")
            
            st.markdown("---")
            
            # Make predictions
            st.subheader("🤖 Making Predictions")
            
            # Batch size selector for large files
            batch_size = st.slider(
                "Batch size for predictions", 
                min_value=100, 
                max_value=10000, 
                value=1000,
                help="Process predictions in batches to manage memory"
            )
            
            if st.button("🚀 Run Predictions", type="primary"):
                with st.spinner(f"Making predictions on {len(processed_df)} rows..."):
                    try:
                        # Make predictions in batches
                        all_predictions = []
                        all_probabilities = []
                        
                        for i in range(0, len(processed_df), batch_size):
                            batch = processed_df.iloc[i:i+batch_size]
                            
                            # Predict
                            preds = model.predict(batch)
                            all_predictions.extend(preds)
                            
                            # Get probabilities if available
                            if hasattr(model, 'predict_proba'):
                                proba = model.predict_proba(batch)
                                conf = [max(p) for p in proba]
                                all_probabilities.extend(conf)
                            else:
                                all_probabilities.extend([0.95] * len(preds))
                            
                            # Show progress
                            progress = min(100, int((i + len(batch)) / len(processed_df) * 100))
                            st.progress(progress / 100, text=f"Progress: {progress}%")
                        
                        # Add predictions to dataframe
                        results_df = processed_df.copy()
                        results_df['prediction'] = all_predictions
                        results_df['confidence'] = all_probabilities
                        results_df['fault_status'] = results_df['prediction'].map({0: 'NORMAL', 1: 'FAULT'})
                        
                        st.success(f"✅ Successfully made {len(results_df)} predictions!")
                        
                        # Show results summary
                        st.subheader("📊 Results Summary")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Predictions", len(results_df))
                        with col2:
                            faults = results_df['prediction'].sum()
                            st.metric("Faults Detected", int(faults))
                        with col3:
                            fault_rate = (faults / len(results_df)) * 100
                            st.metric("Fault Rate", f"{fault_rate:.2f}%")
                        with col4:
                            avg_conf = results_df['confidence'].mean()
                            st.metric("Avg Confidence", f"{avg_conf:.1%}")
                        
                        # Show results table
                        with st.expander("📋 View Predictions", expanded=True):
                            display_cols = ['prediction', 'fault_status', 'confidence']
                            # Add first few sensor columns
                            sensor_cols = [c for c in results_df.columns if '_' in c and c not in display_cols][:5]
                            display_cols.extend(sensor_cols)
                            
                            st.dataframe(
                                results_df[display_cols].head(100),
                                use_container_width=True
                            )
                            if len(results_df) > 100:
                                st.caption(f"Showing first 100 of {len(results_df)} rows")
                        
                        # Download results
                        st.subheader("💾 Download Results")
                        
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Predictions as CSV",
                            data=csv,
                            file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                        
                        # Option to save to MongoDB
                        st.subheader("💽 Save to Database")
                        if st.button("Save Predictions to MongoDB"):
                            try:
                                client = MongoDBClient()
                                db = client.database
                                collection = db['predictions']
                                
                                # Create records
                                records = []
                                for i in range(len(results_df)):
                                    record = {
                                        'timestamp': datetime.now(),
                                        'prediction': int(results_df.iloc[i]['prediction']),
                                        'confidence': float(results_df.iloc[i]['confidence']),
                                        'reviewed': False
                                    }
                                    # Add some sensor values
                                    for col in sensor_cols[:5]:
                                        if col in results_df.columns:
                                            record[col] = float(results_df.iloc[i][col])
                                    records.append(record)
                                
                                collection.insert_many(records)
                                st.success(f"✅ Saved {len(records)} predictions to MongoDB!")
                            except Exception as e:
                                st.error(f"❌ Error saving to MongoDB: {e}")
                    
                    except Exception as e:
                        st.error(f"❌ Error making predictions: {e}")
                        import traceback
                        st.code(traceback.format_exc())
        
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")
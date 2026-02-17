import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta

sys.path.append(str(Path(__file__).parent.parent.parent))

from sensor.ml.model.estimator import ModelResolver
from sensor.utils.main_utils import load_object
from sensor.constant.training_pipeline import SAVED_MODEL_DIR
from sensor.configuration.mongo_db_connection import MongoDBClient

def show():
    st.title("# 📊 Model Insights")
    # No title here - title comes from app.py
    
    # Get absolute path to saved_models
    project_root = Path(__file__).parent.parent.parent
    model_dir = os.path.join(project_root, "saved_models")
    
    # Check for trained model
    try:
        if not os.path.exists(model_dir):
            st.error(f"Directory not found: {model_dir}")
            return
            
        # List available model versions
        model_versions = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
        
        if not model_versions:
            st.warning("No model versions found")
            return
            
        st.success(f"Found {len(model_versions)} model version(s)")
        
        # Let user select model version
        selected_version = st.selectbox("Select Model Version", model_versions)
        
        if selected_version:
            model_path = os.path.join(model_dir, selected_version, "model.pkl")
            
            if os.path.exists(model_path):
                model = load_object(model_path)
                
                col1, col2 = st.columns(2)
                with col1:
                    if hasattr(model, 'model'):
                        st.info(f"Model Type: {type(model.model).__name__}")
                with col2:
                    st.info(f"Version: {selected_version}")
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
    
    # Get recent predictions
    try:
        client = MongoDBClient()
        db = client.database
        
        if 'predictions' in db.list_collection_names():
            data = list(db.predictions.find().sort('timestamp', -1).limit(1000))
            
            if data:
                df = pd.DataFrame(data)
                
                if 'prediction' in df.columns:
                    total = len(df)
                    faults = df['prediction'].sum()
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total Predictions", f"{total:,}")
                    col2.metric("Faults Detected", f"{int(faults):,}")
                    
                    if 'confidence' in df.columns:
                        col3.metric("Avg Confidence", f"{df['confidence'].mean():.1%}")
                    
                    col4.metric("Fault Rate", f"{(faults/total*100):.1f}%")
    
    except Exception as e:
        st.warning(f"Could not load prediction metrics")

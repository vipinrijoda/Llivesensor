import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import sys
import os
import time
from pathlib import Path
from datetime import datetime, timedelta

sys.path.append(str(Path(__file__).parent.parent.parent))

from sensor.ml.model.estimator import ModelResolver
from sensor.utils.main_utils import load_object
from sensor.constant.training_pipeline import SAVED_MODEL_DIR
from sensor.configuration.mongo_db_connection import MongoDBClient

def show():
    st.title("📊 Model Insights")
    
    # Get absolute path to saved_models
    project_root = Path(__file__).parent.parent.parent
    model_dir = os.path.join(project_root, "saved_models")
    
    # Check if directory exists (Git LFS might still be downloading)
    if not os.path.exists(model_dir):
        st.warning("⏳ Model directory is being downloaded from Git LFS. Please wait...")
        
        # Show progress simulation
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(10):
            time.sleep(0.3)  # Simulate waiting
            progress_bar.progress((i + 1) * 10)
            status_text.text(f"Downloading model files... {(i + 1) * 10}%")
        
        progress_bar.empty()
        status_text.empty()
        
        st.info("""
        **Why this happens:**
        - Your model is stored in Git LFS (Large File Storage)
        - Streamlit Cloud downloads LFS files after the initial clone
        - This can take 1-5 minutes depending on file size
        
        **Click the button below to check again:**
        """)
        
        if st.button("🔄 Check Again", type="primary"):
            st.rerun()
        return
    
    # Check if directory exists but is empty
    if os.path.exists(model_dir) and not os.listdir(model_dir):
        st.warning("⏳ Model directory found but files are still downloading. Please wait...")
        if st.button("🔄 Refresh"):
            st.rerun()
        return
    
    # Check for trained model
    try:
        # List available model versions
        model_versions = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
        
        if not model_versions:
            st.warning("⚠️ No model versions found. Please wait for Git LFS to complete download.")
            st.info("This may take a few minutes. Refresh the page to check again.")
            if st.button("🔄 Refresh Page"):
                st.rerun()
            return
            
        st.success(f"✅ Found {len(model_versions)} model version(s)")
        
        # Let user select model version
        selected_version = st.selectbox("Select Model Version", model_versions)
        
        if selected_version:
            model_path = os.path.join(model_dir, selected_version, "model.pkl")
            
            # Check if model file exists
            if not os.path.exists(model_path):
                st.error(f"❌ Model file not found at: {model_path}")
                st.info("This might happen if Git LFS hasn't finished downloading. Please wait and refresh.")
                if st.button("🔄 Refresh Page", key="refresh_model"):
                    st.rerun()
                return
            
            # Load the model
            with st.spinner("Loading model..."):
                try:
                    model = load_object(model_path)
                    st.success("✅ Model loaded successfully!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if hasattr(model, 'model'):
                            st.info(f"**Model Type:** {type(model.model).__name__}")
                    with col2:
                        st.info(f"**Version:** {selected_version}")
                    
                    # Try to get feature info
                    if hasattr(model, 'preprocessor'):
                        if hasattr(model.preprocessor, 'feature_names_in_'):
                            n_features = len(model.preprocessor.feature_names_in_)
                            st.info(f"**Number of Features:** {n_features}")
                            
                            # Show sample features
                            with st.expander("🔍 View Sample Features"):
                                features = model.preprocessor.feature_names_in_.tolist()
                                st.write(features[:20])
                                if len(features) > 20:
                                    st.write(f"... and {len(features)-20} more")
                
                except Exception as e:
                    st.error(f"❌ Error loading model: {e}")
                    return
    
    except Exception as e:
        st.error(f"Error accessing model directory: {str(e)}")
        return
    
    st.markdown("---")
    
    # Get recent predictions
    st.subheader("📈 Prediction Statistics")
    
    try:
        client = MongoDBClient()
        db = client.database
        
        if 'predictions' not in db.list_collection_names():
            st.info("ℹ️ No predictions collection found in database.")
            st.info("Run predictions first to see statistics.")
        else:
            total = db.predictions.count_documents({})
            
            if total == 0:
                st.info("ℹ️ No predictions found in database.")
                st.info("Run predictions first to see statistics.")
            else:
                # Get fault count
                faults = db.predictions.count_documents({"prediction": 1})
                
                # Get average confidence
                pipeline = [{
                    "$group": {
                        "_id": None,
                        "avg_conf": {"$avg": "$confidence"},
                        "min_conf": {"$min": "$confidence"},
                        "max_conf": {"$max": "$confidence"}
                    }
                }]
                
                result = list(db.predictions.aggregate(pipeline))
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Predictions", f"{total:,}")
                col2.metric("Faults Detected", f"{faults:,}")
                col3.metric("Fault Rate", f"{(faults/total*100):.2f}%")
                
                if result and result[0]['avg_conf']:
                    col4.metric("Avg Confidence", f"{result[0]['avg_conf']:.2%}")
                
                # Get recent predictions for charts
                data = list(db.predictions.find().sort('timestamp', -1).limit(1000))
                
                if data:
                    df = pd.DataFrame(data)
                    
                    # Create tabs for different visualizations
                    tab1, tab2, tab3 = st.tabs(["📊 Distribution", "📈 Over Time", "🔍 Recent"])
                    
                    with tab1:
                        # Prediction distribution pie chart
                        fig = px.pie(
                            values=[total - faults, faults],
                            names=['Normal', 'Fault'],
                            title="Prediction Distribution",
                            color_discrete_sequence=['green', 'red']
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Confidence distribution if available
                        if 'confidence' in df.columns:
                            fig = px.histogram(
                                df,
                                x='confidence',
                                nbins=20,
                                title="Confidence Distribution",
                                labels={'confidence': 'Confidence Score'}
                            )
                            fig.add_vline(x=0.7, line_dash="dash", line_color="orange", annotation_text="Threshold")
                            fig.add_vline(x=0.9, line_dash="dash", line_color="green", annotation_text="High Confidence")
                            st.plotly_chart(fig, use_container_width=True)
                    
                    with tab2:
                        # Predictions over time
                        if 'timestamp' in df.columns:
                            df['date'] = pd.to_datetime(df['timestamp']).dt.date
                            daily = df.groupby('date').agg({
                                'prediction': ['sum', 'count']
                            }).reset_index()
                            daily.columns = ['date', 'faults', 'total']
                            daily['fault_rate'] = (daily['faults'] / daily['total'] * 100)
                            
                            fig = go.Figure()
                            fig.add_trace(go.Bar(
                                x=daily['date'],
                                y=daily['faults'],
                                name='Daily Faults',
                                marker_color='red'
                            ))
                            fig.add_trace(go.Scatter(
                                x=daily['date'],
                                y=daily['fault_rate'],
                                name='Fault Rate %',
                                yaxis='y2',
                                line=dict(color='blue', width=2)
                            ))
                            
                            fig.update_layout(
                                title="Daily Faults and Fault Rate",
                                xaxis_title="Date",
                                yaxis_title="Number of Faults",
                                yaxis2=dict(
                                    title="Fault Rate (%)",
                                    overlaying='y',
                                    side='right',
                                    range=[0, max(daily['fault_rate'].max() * 1.1, 10)]
                                ),
                                height=400
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    
                    with tab3:
                        # Recent predictions table
                        st.write("**Recent Predictions**")
                        display_df = df.head(10).copy()
                        if 'timestamp' in display_df.columns:
                            display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
                        
                        cols_to_show = ['timestamp', 'prediction', 'confidence']
                        available_cols = [c for c in cols_to_show if c in display_df.columns]
                        st.dataframe(display_df[available_cols], use_container_width=True)
    
    except Exception as e:
        st.warning(f"Could not load prediction metrics: {e}")
    
    # Footer
    st.markdown("---")
    st.caption("Model insights updated in real-time from database")

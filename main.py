import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
import time
from pathlib import Path
from datetime import datetime, timedelta
import traceback

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from sensor.configuration.mongo_db_connection import MongoDBClient
from sensor.ml.model.estimator import ModelResolver
from sensor.utils.main_utils import load_object
from sensor.constant.training_pipeline import SAVED_MODEL_DIR
from sensor.pipeline.training_pipeline import TrainPipeline

# ============================================
# YOUR DATABASE CONFIGURATION
# ============================================
DATABASE_NAME = "aps_fault_sensor"
COLLECTION_NAME = "sensor"
PREDICTIONS_COLLECTION = "predictions"

# ============================================
# Page Configuration
# ============================================
st.set_page_config(
    page_title="APS Sensor Dashboard",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# Constants
# ============================================
COLUMNS_TO_DROP = ['br_000', 'bq_000', 'bp_000', 'ab_000', 'cr_000', 'bo_000', 'bn_000']

# All sensor columns (simplified list - add all your columns)
ALL_SENSOR_COLUMNS = [
    'aa_000', 'ac_000', 'ad_000', 'ae_000', 'af_000', 'ag_000', 'ag_001', 
    'ag_002', 'ag_003', 'ag_004', 'ag_005', 'ag_006', 'ag_007', 'ag_008', 
    'ag_009', 'ah_000', 'ai_000', 'aj_000', 'ak_000', 'al_000', 'am_0', 
    'an_000', 'ao_000', 'ap_000', 'aq_000', 'ar_000', 'as_000', 'at_000', 
    'au_000', 'av_000', 'ax_000', 'ay_000', 'ay_001', 'ay_002', 'ay_003', 
    'ay_004', 'ay_005', 'ay_006', 'ay_007', 'ay_008', 'ay_009', 'az_000', 
    'az_001', 'az_002', 'az_003', 'az_004', 'az_005', 'az_006', 'az_007', 
    'az_008', 'az_009', 'ba_000', 'ba_001', 'ba_002', 'ba_003', 'ba_004', 
    'ba_005', 'ba_006', 'ba_007', 'ba_008', 'ba_009', 'bb_000', 'bc_000', 
    'bd_000', 'be_000', 'bf_000', 'bg_000', 'bh_000', 'bi_000', 'bj_000', 
    'bk_000', 'bl_000', 'bm_000', 'bs_000', 'bt_000', 'bu_000', 'bv_000', 
    'bx_000', 'by_000', 'bz_000', 'ca_000', 'cb_000', 'cc_000', 'cd_000', 
    'ce_000', 'cf_000', 'cg_000', 'ch_000', 'ci_000', 'cj_000', 'ck_000', 
    'cl_000', 'cm_000', 'cn_000', 'cn_001', 'cn_002', 'cn_003', 'cn_004', 
    'cn_005', 'cn_006', 'cn_007', 'cn_008', 'cn_009', 'co_000', 'cp_000', 
    'cq_000', 'cs_000', 'cs_001', 'cs_002', 'cs_003', 'cs_004', 'cs_005', 
    'cs_006', 'cs_007', 'cs_008', 'cs_009', 'ct_000', 'cu_000', 'cv_000', 
    'cx_000', 'cy_000', 'cz_000', 'da_000', 'db_000', 'dc_000', 'dd_000', 
    'de_000', 'df_000', 'dg_000', 'dh_000', 'di_000', 'dj_000', 'dk_000', 
    'dl_000', 'dm_000', 'dn_000', 'do_000', 'dp_000', 'dq_000', 'dr_000', 
    'ds_000', 'dt_000', 'du_000', 'dv_000', 'dx_000', 'dy_000', 'dz_000', 
    'ea_000', 'eb_000', 'ec_00', 'ed_000', 'ee_000', 'ee_001', 'ee_002', 
    'ee_003', 'ee_004', 'ee_005', 'ee_006', 'ee_007', 'ee_008', 'ee_009', 
    'ef_000', 'eg_000'
]

# ============================================
# Helper Functions
# ============================================
def format_timestamp(ts):
    """Format timestamp to show only up to seconds (no milliseconds)"""
    if ts is None:
        return 'Unknown'
    if isinstance(ts, (pd.Timestamp, datetime)):
        return ts.strftime('%Y-%m-%d %H:%M:%S')
    if isinstance(ts, str):
        try:
            parsed_ts = pd.to_datetime(ts)
            return parsed_ts.strftime('%Y-%m-%d %H:%M:%S')
        except:
            return ts
    return str(ts)

def format_indian_currency(amount):
    """Format amount in Indian Rupees with lakhs/crores notation"""
    if amount >= 10000000:  # 1 Crore+
        return f"₹{amount/10000000:.2f} Cr"
    elif amount >= 100000:  # 1 Lakh+
        return f"₹{amount/100000:.2f} L"
    else:
        return f"₹{amount:,.0f}"

def check_model_exists():
    """Check if a trained model exists"""
    model_dir = Path("saved_models")
    if not model_dir.exists():
        return False
    versions = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
    return len(versions) > 0

def get_model_versions():
    """Get list of available model versions"""
    model_dir = Path("saved_models")
    if not model_dir.exists():
        return []
    return [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]

def train_model():
    """Run training pipeline"""
    try:
        with st.spinner("Training in progress... This may take 5-10 minutes."):
            pipeline = TrainPipeline()
            pipeline.run_pipeline()
        return True, "Training completed successfully!"
    except Exception as e:
        return False, f"Training failed: {str(e)}"

def get_sensor_data(limit=10000):
    """
    Get REAL sensor data from MongoDB - NO SAMPLE DATA
    Returns None if no data found
    """
    try:
        client = MongoDBClient(database_name=DATABASE_NAME)
        db = client.database
        
        # Check if sensor collection exists
        if COLLECTION_NAME not in db.list_collection_names():
            return None
        
        collection = db[COLLECTION_NAME]
        count = collection.count_documents({})
        
        if count == 0:
            return None
        
        data = list(collection.find().limit(limit))
        if data:
            df = pd.DataFrame(data)
            if '_id' in df.columns:
                df = df.drop('_id', axis=1)
            return df
        return None
    except Exception as e:
        return None

def get_prediction_data(limit=None, days=None):
    """
    Get REAL prediction data from MongoDB - NO SAMPLE DATA
    If limit is None, returns ALL data
    """
    try:
        client = MongoDBClient(database_name=DATABASE_NAME)
        db = client.database
        
        # Check if predictions collection exists
        if PREDICTIONS_COLLECTION not in db.list_collection_names():
            return None
        
        collection = db[PREDICTIONS_COLLECTION]
        
        # Build query
        query = {}
        if days:
            start_date = datetime.now() - timedelta(days=days)
            query['timestamp'] = {'$gte': start_date}
        
        # Get data - NO LIMIT if limit is None
        cursor = collection.find(query).sort('timestamp', -1)
        if limit is not None:
            cursor = cursor.limit(limit)
        
        data = list(cursor)
        
        if not data:
            return None
        
        df = pd.DataFrame(data)
        if '_id' in df.columns:
            df = df.drop('_id', axis=1)
        
        # Ensure timestamp is datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
        
    except Exception as e:
        return None

def save_predictions_to_db(predictions_df, max_records=50):
    """Save prediction results to MongoDB with proper error handling"""
    try:
        client = MongoDBClient(database_name=DATABASE_NAME)
        db = client.database
        collection = db[PREDICTIONS_COLLECTION]
        
        records = []
        for i in range(min(len(predictions_df), max_records)):
            # Safely get prediction value
            pred_val = predictions_df.iloc[i]['prediction']
            if isinstance(pred_val, (np.integer, np.floating)):
                pred_int = int(pred_val)
            else:
                pred_int = int(pred_val) if pred_val else 0
            
            # Safely get confidence value
            if 'confidence' in predictions_df.columns:
                conf_val = predictions_df.iloc[i]['confidence']
                if isinstance(conf_val, (np.floating, float)):
                    conf_float = float(conf_val)
                else:
                    conf_float = float(conf_val) if conf_val else 0.95
            else:
                conf_float = 0.95
            
            # Create base record
            record = {
                'timestamp': datetime.now(),
                'prediction': pred_int,
                'confidence': conf_float,
                'reviewed': False,
                'batch_id': datetime.now().strftime('%Y%m%d_%H%M%S')
            }
            
            # Add first 5 sensor values safely
            sensor_cols = [c for c in predictions_df.columns if '_' in c and c not in ['prediction', 'confidence', 'fault_status', 'timestamp']]
            for j, col in enumerate(sensor_cols[:5]):
                if col in predictions_df.columns:
                    try:
                        val = predictions_df.iloc[i][col]
                        if pd.isna(val):
                            record[col] = 0.0
                        elif isinstance(val, (np.integer, np.floating)):
                            record[col] = float(val)
                        elif isinstance(val, (int, float)):
                            record[col] = float(val)
                        else:
                            try:
                                record[col] = float(val)
                            except:
                                record[col] = 0.0
                    except:
                        record[col] = 0.0
            
            records.append(record)
        
        if records:
            result = collection.insert_many(records)
            return True, len(result.inserted_ids)
        return False, 0
        
    except Exception as e:
        return False, str(e)

# ============================================
# Session State Initialization
# ============================================
if 'training_in_progress' not in st.session_state:
    st.session_state.training_in_progress = False
if 'training_complete' not in st.session_state:
    st.session_state.training_complete = False
if 'last_training_time' not in st.session_state:
    st.session_state.last_training_time = None

# ============================================
# Sidebar
# ============================================
with st.sidebar:
    st.title("🔧 APS Sensor")
    st.markdown("---")
    
    # Navigation - REMOVED LIVE MONITORING
    page = st.radio(
        "Navigation",
        ["🏠 Home", "💰 ROI", "🔮 Predict"]
    )
    
    st.markdown("---")
    
    # Database Status
    st.subheader("📡 Database Status")
    try:
        client = MongoDBClient(database_name=DATABASE_NAME)
        db = client.database
        collections = db.list_collection_names()
        
        st.success(f"✅ Connected to '{DATABASE_NAME}'")
        
        # Show sensor collection
        if COLLECTION_NAME in collections:
            count = db[COLLECTION_NAME].count_documents({})
            st.caption(f"📊 {COLLECTION_NAME}: {count:,} records")
        else:
            st.caption(f"❌ {COLLECTION_NAME}: Not found")
        
        # Show predictions collection
        if PREDICTIONS_COLLECTION in collections:
            count = db[PREDICTIONS_COLLECTION].count_documents({})
            st.caption(f"📊 {PREDICTIONS_COLLECTION}: {count:,} predictions")
            
            # Show latest timestamp if available
            if count > 0:
                latest = db[PREDICTIONS_COLLECTION].find_one(sort=[('timestamp', -1)])
                if latest and 'timestamp' in latest:
                    latest_time = format_timestamp(latest['timestamp'])
                    st.caption(f"🕒 Latest: {latest_time}")
        else:
            st.caption(f"ℹ️ {PREDICTIONS_COLLECTION}: Not found")
            
    except Exception as e:
        st.error(f"❌ MongoDB Error")
        st.caption(str(e)[:50])
    
    st.markdown("---")
    
    # Model Status
    st.subheader("🤖 Model Status")
    if check_model_exists():
        st.success("✅ Model trained")
        versions = get_model_versions()
        st.caption(f"📁 {len(versions)} version(s)")
    else:
        st.warning("⚠️ No model")

# ============================================
# HOME PAGE - WITH DATABASE SUMMARY & COLUMN DETAILS
# ============================================
if page == "🏠 Home":
    st.title("🏠 APS Sensor Fault Detection")
    
    # Get sensor data for summary
    sensor_df = get_sensor_data(limit=1000)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        ### About
        This system predicts **Air Pressure System (APS)** failures in heavy trucks.
        
        **Database:** `{DATABASE_NAME}`
        **Collections:** `{COLLECTION_NAME}`, `{PREDICTIONS_COLLECTION}`
        
        **Features:**
        - 💰 **ROI Calculator** - See business impact in Indian Rupees
        - 🔮 **Batch Prediction** - Upload CSV for predictions
        """)
        
        # Training Section
        st.markdown("### 🏋️ Model Training")
        
        if st.button("🚀 Train New Model", disabled=st.session_state.training_in_progress):
            st.session_state.training_in_progress = True
            success, message = train_model()
            st.session_state.training_in_progress = False
            st.session_state.training_complete = success
            st.session_state.last_training_time = datetime.now()
            
            if success:
                st.success(message)
                st.balloons()
            else:
                st.error(message)
        
        if st.session_state.training_in_progress:
            st.warning("Training in progress... Please wait.")
        
        if st.session_state.training_complete and st.session_state.last_training_time:
            st.info(f"Last training: {st.session_state.last_training_time.strftime('%Y-%m-%d %H:%M')}")
    
    with col2:
        # Database Statistics
        st.markdown("### 📊 Database Statistics")
        
        # Get prediction stats
        pred_df = get_prediction_data(limit=1000)
        
        if pred_df is not None and not pred_df.empty:
            total = len(pred_df)
            faults = pred_df['prediction'].sum() if 'prediction' in pred_df.columns else 0
            
            st.metric(f"📊 {PREDICTIONS_COLLECTION}", f"{total:,} predictions")
            st.metric("⚠️ Faults Detected", f"{int(faults):,}")
            if total > 0:
                st.metric("📈 Fault Rate", f"{(faults/total*100):.1f}%")
            
            if 'confidence' in pred_df.columns:
                st.metric("🎯 Avg Confidence", f"{pred_df['confidence'].mean():.1%}")
        else:
            st.info(f"No data in '{PREDICTIONS_COLLECTION}' collection")
    
    # Database Summary Section
    st.markdown("---")
    st.subheader("📋 Database Schema & Column Details")
    
    if sensor_df is not None and not sensor_df.empty:
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["📊 Data Preview", "🔧 Column Info", "📈 Statistics"])
        
        with tab1:
            st.markdown("### Sample Data (First 10 rows)")
            display_df = sensor_df.head(10).copy()
            st.dataframe(display_df, use_container_width=True)
        
        with tab2:
            st.markdown("### Column Information")
            
            # Get column info
            col_info = []
            for col in sensor_df.columns:
                col_type = sensor_df[col].dtype
                missing = sensor_df[col].isna().sum()
                missing_pct = (missing / len(sensor_df)) * 100
                
                if col in ALL_SENSOR_COLUMNS:
                    category = "Sensor"
                elif col == 'class':
                    category = "Target Variable"
                else:
                    category = "Other"
                
                col_info.append({
                    "Column": col,
                    "Type": str(col_type),
                    "Category": category,
                    "Missing Values": missing,
                    "Missing %": f"{missing_pct:.1f}%"
                })
            
            col_df = pd.DataFrame(col_info)
            st.dataframe(col_df, use_container_width=True)
            
            # Column categories summary
            st.markdown("### Column Categories")
            cat_counts = col_df['Category'].value_counts()
            for cat, count in cat_counts.items():
                st.write(f"- **{cat}:** {count} columns")
        
        with tab3:
            st.markdown("### Statistical Summary")
            
            # Numeric columns statistics
            numeric_cols = sensor_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                stats_df = sensor_df[numeric_cols].describe().T
                st.dataframe(stats_df, use_container_width=True)
            
            # Class distribution if exists
            if 'class' in sensor_df.columns:
                st.markdown("### Class Distribution")
                class_counts = sensor_df['class'].value_counts()
                fig = px.pie(
                    values=class_counts.values,
                    names=class_counts.index,
                    title="Class Distribution",
                    color_discrete_sequence=['green', 'red']
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"No data found in '{COLLECTION_NAME}' collection")

# ============================================
# ROI PAGE - INDIAN RUPEES
# ============================================
elif page == "💰 ROI":
    st.title("💰 ROI Calculator (₹ Indian Rupees)")
    
    # Parameters (converted to ₹)
    with st.expander("⚙️ Parameters", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            cost_per_fault = st.number_input("Cost per Fault (₹)", 100000, 10000000, 4000000, 100000)
            saving_per_fault = st.number_input("Savings per Prevention (₹)", 100000, 10000000, 2500000, 100000)
        with col2:
            false_alarm_cost = st.number_input("False Alarm Cost (₹)", 1000, 50000, 10000, 1000)
            system_cost = st.number_input("Annual System Cost (₹)", 100000, 2000000, 800000, 50000)
    
    # Get REAL prediction data
    df = get_prediction_data(limit=None)
    
    if df is not None and not df.empty and 'prediction' in df.columns:
        total = len(df)
        faults = int(df['prediction'].sum())
        
        # Calculations based on REAL data
        false_positives = int(total * 0.05 * 0.5)  # Estimate
        savings = faults * saving_per_fault
        costs = false_positives * false_alarm_cost + system_cost/12
        net = savings - costs
        roi = (net / costs) * 100 if costs > 0 else 0
        
        # Display metrics in Indian format
        st.subheader(f"📊 ROI Summary (Based on {total:,} predictions)")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Predictions", f"{total:,}")
        with col2:
            st.metric("Faults Detected", f"{faults:,}")
        with col3:
            st.metric("Est. False Positives", f"{false_positives:,}")
        with col4:
            st.metric("ROI", f"{roi:.1f}%")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Monthly Savings", format_indian_currency(savings))
        with col2:
            st.metric("Monthly Costs", format_indian_currency(costs))
        with col3:
            st.metric("Net Savings", format_indian_currency(net))
        
        # Chart with Indian currency in hover
        labels = ['Prevention Savings', 'False Alarm Costs', 'System Cost']
        values = [savings, false_positives * false_alarm_cost, system_cost/12]
        
        fig = px.pie(
            values=values, 
            names=labels, 
            title="Monthly Cost Breakdown (₹ Indian Rupees)",
            hole=0.3
        )
        
        # Customize hover template to show Indian currency
        fig.update_traces(
            hovertemplate='<b>%{label}</b><br>Amount: ₹%{value:,.0f}<br>Percentage: %{percent}<extra></extra>'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Additional details
        with st.expander("📋 Detailed Calculations"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Assumptions:**")
                st.write(f"- Detection accuracy: 95%")
                st.write(f"- False positive rate: 2.5% of total predictions")
                st.write(f"- Based on {total} total predictions")
            
            with col2:
                st.markdown("**Breakdown:**")
                st.write(f"- Prevention Savings: {format_indian_currency(savings)}")
                st.write(f"- False Alarm Costs: {format_indian_currency(false_positives * false_alarm_cost)}")
                st.write(f"- System Cost (monthly): {format_indian_currency(system_cost/12)}")
    else:
        st.warning(f"No REAL data in '{PREDICTIONS_COLLECTION}' for ROI calculation")

# ============================================
# PREDICT PAGE
# ============================================
elif page == "🔮 Predict":
    st.title("🔮 Batch Prediction")
    
    # Check for model
    if not check_model_exists():
        st.warning("⚠️ No trained model found. Please train a model first on the Home page.")
        st.stop()
    
    # Get model versions
    versions = get_model_versions()
    if not versions:
        st.warning("No model versions found")
        st.stop()
    
    selected_version = st.selectbox("Select Model Version", versions)
    model_path = Path("saved_models") / selected_version / "model.pkl"
    
    try:
        with st.spinner("Loading model..."):
            model = load_object(model_path)
        
        st.success("✅ Model loaded")
        
        # Get expected features
        expected_features = None
        if hasattr(model, 'preprocessor') and hasattr(model.preprocessor, 'feature_names_in_'):
            expected_features = model.preprocessor.feature_names_in_.tolist()
            st.info(f"Model expects {len(expected_features)} features")
        
        # File upload
        uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.success(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
            
            with st.expander("Preview"):
                st.dataframe(df.head())
            
            # Preprocessing
            processed_df = df.copy()
            
            # Drop columns
            dropped = []
            for col in COLUMNS_TO_DROP:
                if col in processed_df.columns:
                    processed_df = processed_df.drop(col, axis=1)
                    dropped.append(col)
            
            if dropped:
                st.write(f"Dropped: {', '.join(dropped)}")
            
            # Remove class column
            if 'class' in processed_df.columns:
                processed_df = processed_df.drop('class', axis=1)
            
            # Convert to numeric
            for col in processed_df.columns:
                processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
            
            processed_df = processed_df.fillna(0)
            
            # Align features
            if expected_features:
                missing = []
                for col in expected_features:
                    if col not in processed_df.columns:
                        processed_df[col] = 0
                        missing.append(col)
                
                if missing:
                    st.warning(f"Added {len(missing)} missing features with default values")
                
                processed_df = processed_df[expected_features]
                st.success(f"Data ready: {processed_df.shape[1]} features")
            
            # Batch size
            batch_size = st.slider("Batch Size", 100, 5000, 1000)
                        
            if st.button("Run Predictions", type="primary"):

                progress = st.progress(0)
                all_preds = []
                all_conf = []

                for i in range(0, len(processed_df), batch_size):
                    batch = processed_df.iloc[i:i+batch_size]

                    preds = model.predict(batch)
                    all_preds.extend(preds)

                    # Confidence Handling
                    if hasattr(model, "predict_proba"):
                        probs = model.predict_proba(batch)
                        conf = probs.max(axis=1)
                        all_conf.extend(conf)
                    else:
                        all_conf.extend([0.95] * len(batch))

                    progress.progress(min(1.0, (i + len(batch)) / len(processed_df)))

                # =====================================
                # CREATE RESULTS DATAFRAME
                # =====================================

                results_df = df.copy()
                results_df['prediction'] = all_preds
                results_df['confidence'] = all_conf
                results_df['timestamp'] = datetime.now()

                # Human readable labels
                results_df['fault_status'] = results_df['prediction'].apply(
                    lambda x: "⚠️ Fault" if x == 1 else "✅ Normal"
                )

                # Severity Logic
                def get_severity(row):
                    if row['prediction'] == 1 and row['confidence'] >= 0.90:
                        return "🔴 Critical"
                    elif row['prediction'] == 1:
                        return "🟠 High"
                    else:
                        return "🟢 Normal"

                results_df['severity'] = results_df.apply(get_severity, axis=1)

                # =====================================
                # SUMMARY METRICS
                # =====================================

                total = len(results_df)
                faults = int(results_df['prediction'].sum())
                fault_rate = (faults / total) * 100 if total > 0 else 0

                st.subheader("📊 Prediction Summary")

                col1, col2, col3, col4 = st.columns(4)

                col1.metric("Total Records", f"{total:,}")
                col2.metric("Faults Detected", f"{faults:,}")
                col3.metric("Fault Rate", f"{fault_rate:.1f}%")
                col4.metric("Avg Confidence", f"{results_df['confidence'].mean()*100:.1f}%")

                # =====================================
                # VISUALIZATION
                # =====================================

                st.subheader("📈 Prediction Distribution")

                fig = px.pie(
                    results_df,
                    names='fault_status',
                    hole=0.4,
                    title="Fault vs Normal Distribution"
                )

                fig.update_traces(
                    hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>"
                )

                st.plotly_chart(fig, use_container_width=True)

                # =====================================
                # TOP FAULTS SECTION
                # =====================================

                if faults > 0:
                    st.subheader("🚨 Top Critical Faults")

                    critical_df = results_df[results_df['severity'] == "🔴 Critical"]
                    critical_df = critical_df.sort_values(by="confidence", ascending=False)

                    if not critical_df.empty:
                        st.dataframe(
                            critical_df[['fault_status', 'severity', 'confidence', 'timestamp']].head(10),
                            use_container_width=True
                        )
                    else:
                        st.info("No critical faults detected.")

                # =====================================
                # DETAILED RESULTS TABLE
                # =====================================

                st.subheader("📋 Detailed Results")

                display_cols = ['fault_status', 'severity', 'confidence', 'timestamp']

                formatted_df = results_df[display_cols].copy()
                formatted_df['confidence'] = formatted_df['confidence'].apply(lambda x: f"{x*100:.2f}%")
                formatted_df['timestamp'] = formatted_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

                st.dataframe(
                    formatted_df,
                    use_container_width=True,
                    height=450
                )

                # =====================================
                # DOWNLOAD BUTTON
                # =====================================

                csv = results_df.to_csv(index=False)

                st.download_button(
                    "📥 Download Full Prediction Results",
                    csv,
                    f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv"
                )

                # =====================================
                # SAVE TO DATABASE
                # =====================================

                if st.button("💾 Save to Database"):
                    success, result = save_predictions_to_db(results_df)
                    if success:
                        st.success(f"✅ Saved {result} predictions to '{PREDICTIONS_COLLECTION}' collection")
                    else:
                        st.error(f"❌ Error saving: {result}")
    
    except Exception as e:
        st.error(f"Error: {e}")
        st.code(traceback.format_exc())

# ============================================
# Footer
# ============================================
st.markdown("---")
st.caption(f"APS Sensor Fault Detection System | Database: {DATABASE_NAME} | Currency: ₹ Indian Rupees")




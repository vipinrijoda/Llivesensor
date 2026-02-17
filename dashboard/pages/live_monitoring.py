import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sys
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np

sys.path.append(str(Path(__file__).parent.parent.parent))

from sensor.configuration.mongo_db_connection import MongoDBClient

# YOUR ACTUAL SENSOR COLUMNS FROM SCHEMA.YAML
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

def format_timestamp(ts):
    """Helper function to format timestamp without milliseconds"""
    if ts is None or ts == 'Unknown':
        return 'Unknown'
    if isinstance(ts, (pd.Timestamp, datetime)):
        return ts.strftime('%Y-%m-%d %H:%M:%S')
    if isinstance(ts, str):
        try:
            # Try to parse string timestamp
            parsed_ts = pd.to_datetime(ts)
            return parsed_ts.strftime('%Y-%m-%d %H:%M:%S')
        except:
            return ts  # Return as is if parsing fails
    return str(ts)

def show():
    st.title(" Live Monitoring") 
    
    try:
        # Connect to MongoDB
        client = MongoDBClient()
        db = client.database
        
        # Check if predictions collection exists
        if 'predictions' not in db.list_collection_names():
            st.warning("⚠️ No 'predictions' collection found in database")
            st.info("Please run the training pipeline to generate predictions")
            return
        
        # Get recent predictions (last 24 hours)
        last_24h = datetime.now() - timedelta(days=1)
        
        data = list(db.predictions.find({
            'timestamp': {'$gte': last_24h}
        }).sort('timestamp', -1).limit(500))
        
        if not data:
            st.warning("⚠️ No prediction data found in last 24 hours")
            return
        
        df = pd.DataFrame(data)
        
        # Handle MongoDB ObjectId
        if '_id' in df.columns:
            df['_id'] = df['_id'].astype(str)
        
        # Convert timestamp column to datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        st.success(f"✅ Loaded {len(df)} predictions from last 24 hours")
        
        # Filter to only columns that exist in the dataframe
        available_sensors = [col for col in ALL_SENSOR_COLUMNS if col in df.columns]
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Predictions", len(df))
        
        with col2:
            if 'prediction' in df.columns:
                faults = int(df['prediction'].sum())
                fault_rate = (faults/len(df)*100)
                st.metric("Faults Detected", faults, f"{fault_rate:.1f}%")
            else:
                st.metric("Faults Detected", "N/A")
        
        with col3:
            if 'confidence' in df.columns:
                avg_conf = df['confidence'].mean()
                st.metric("Avg Confidence", f"{avg_conf:.1%}")
            else:
                st.metric("Avg Confidence", "N/A")
        
        with col4:
            st.metric("Active Sensors", len(available_sensors))
        
        # Sensor selector for visualization
        st.subheader("📈 Select Sensors to Visualize")
        
        if available_sensors:
            # Let user select which sensors to view
            selected_sensors = st.multiselect(
                "Choose sensors to display",
                options=available_sensors,
                default=available_sensors[:3] if len(available_sensors) >= 3 else available_sensors
            )
            
            if selected_sensors and 'timestamp' in df.columns:
                # Create sensor chart
                fig = go.Figure()
                
                for sensor in selected_sensors:
                    fig.add_trace(go.Scatter(
                        x=df['timestamp'],
                        y=df[sensor],
                        mode='lines',
                        name=sensor,
                        line=dict(width=2)
                    ))
                
                # Add fault markers if prediction column exists
                if 'prediction' in df.columns:
                    fault_data = df[df['prediction'] == 1]
                    if not fault_data.empty and selected_sensors:
                        # Get max value for y-axis placement
                        y_max = df[selected_sensors].max().max()
                        
                        fig.add_trace(go.Scatter(
                            x=fault_data['timestamp'],
                            y=[y_max * 1.05] * len(fault_data),
                            mode='markers',
                            name='⚠️ Fault',
                            marker=dict(color='red', size=12, symbol='x'),
                            hovertemplate='Fault Detected<br>Time: %{x}<br>Confidence: %{customdata:.1%}<extra></extra>',
                            customdata=fault_data['confidence'] if 'confidence' in fault_data.columns else None
                        ))
                
                fig.update_layout(
                    height=500,
                    xaxis_title="Time",
                    yaxis_title="Sensor Value",
                    hovermode='x unified',
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Statistics for selected sensors
                st.subheader("📊 Sensor Statistics")
                
                stats_data = []
                for sensor in selected_sensors:
                    stats_data.append({
                        'Sensor': sensor,
                        'Mean': f"{df[sensor].mean():.2f}",
                        'Std': f"{df[sensor].std():.2f}",
                        'Min': f"{df[sensor].min():.2f}",
                        'Max': f"{df[sensor].max():.2f}"
                    })
                
                stats_df = pd.DataFrame(stats_data)
                st.dataframe(stats_df, use_container_width=True)
        else:
            st.info("No sensor columns found in the data")
        
        # Recent faults - WITH FIXED TIME FORMAT
        if 'prediction' in df.columns:
            st.subheader("🚨 Recent Faults")
            
            faults_df = df[df['prediction'] == 1].head(10)
            
            if not faults_df.empty:
                for idx, fault in faults_df.iterrows():
                    # Format timestamp using helper function
                    timestamp = format_timestamp(fault.get('timestamp', 'Unknown'))
                    
                    with st.expander(f"⚠️ Fault at {timestamp}"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**Confidence:** {fault.get('confidence', 0):.1%}")
                            st.write(f"**Index:** {idx}")
                        
                        with col2:
                            # Show top 5 sensor values for this fault
                            st.write("**Top Sensor Values:**")
                            if available_sensors:
                                # Get sensors with highest absolute values
                                sensor_values = {s: fault[s] for s in available_sensors[:10] if s in fault}
                                for sensor, value in list(sensor_values.items())[:5]:
                                    st.write(f"- {sensor}: {value:.2f}")
            else:
                st.success("✅ No faults detected in last 24 hours")
        
        # Data quality check - WITH FIXED TIME FORMAT
        with st.expander("🔍 Data Quality Check"):
            st.write("**Missing Values:**")
            missing_data = []
            for sensor in available_sensors[:20]:  # Check first 20 sensors
                missing = df[sensor].isna().sum()
                if missing > 0:
                    missing_data.append({'Sensor': sensor, 'Missing Values': missing, 'Percentage': f"{(missing/len(df)*100):.1f}%"})
            
            if missing_data:
                st.dataframe(pd.DataFrame(missing_data))
            else:
                st.success("✅ No missing values found in sensor data")
            
            # Format time range
            min_time = format_timestamp(df['timestamp'].min())
            max_time = format_timestamp(df['timestamp'].max())
            
            st.write(f"**Total Records:** {len(df)}")
            st.write(f"**Time Range:** {min_time} to {max_time}")
        
        # Raw data view - WITH FIXED TIME FORMAT
        with st.expander("📋 View Raw Data"):
            # Select which columns to show
            cols_to_show = ['timestamp', 'prediction', 'confidence'] + available_sensors[:10]
            cols_to_show = [c for c in cols_to_show if c in df.columns]
            
            # Create a copy to avoid modifying original
            display_df = df[cols_to_show].head(50).copy()
            
            # Format timestamp column
            if 'timestamp' in display_df.columns:
                display_df['timestamp'] = display_df['timestamp'].apply(format_timestamp)
            
            st.dataframe(display_df, use_container_width=True)
            
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
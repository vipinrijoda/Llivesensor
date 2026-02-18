import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sys
import os
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent.parent))

from sensor.ml.model.estimator import ModelResolver
from sensor.utils.main_utils import load_object
from sensor.constant.training_pipeline import SAVED_MODEL_DIR
from sensor.configuration.mongo_db_connection import MongoDBClient
from sensor.constant.database import DATABASE_NAME

# Collection name from your database
COLLECTION_NAME = "sensor"

@st.cache_data
def load_dataset():
    """
    Load dataset from MongoDB using your existing MongoDBClient
    """
    try:
        # Initialize MongoDB client - it will automatically use DATABASE_NAME from your constant
        mongo_client = MongoDBClient()
        db = mongo_client.database
        
        # Check if collection exists
        if COLLECTION_NAME not in db.list_collection_names():
            st.error(f"❌ Collection '{COLLECTION_NAME}' not found in database '{db.name}'")
            st.info("Available collections: " + ", ".join(db.list_collection_names()))
            return None
        
        # Get the collection
        collection = db[COLLECTION_NAME]
        
        # Check if collection has data
        count = collection.count_documents({})
        if count == 0:
            st.warning(f"⚠️ Collection '{COLLECTION_NAME}' is empty")
            return None
        
        # Load data (limit to 1000 rows for performance)
        data = list(collection.find().limit(1000))
        
        if data:
            df = pd.DataFrame(data)
            
            # Remove MongoDB ObjectId if present
            if '_id' in df.columns:
                df = df.drop(columns=['_id'])
            
            st.success(f"✅ Loaded {count:,} records from '{COLLECTION_NAME}' collection")
            st.info(f"📊 Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
            
            return df
        else:
            st.warning("No data found in collection")
            return None
        
    except Exception as e:
        st.error(f"❌ Error loading dataset: {e}")
        return None

# ============================================
# COMPLETE COLUMN DESCRIPTIONS
# ============================================

COLUMN_DESCRIPTIONS = {
    # Target Variable
    'class': {
        'description': 'Target variable indicating APS failure status',
        'values': 'neg = Normal operation, pos = Failure detected',
        'type': 'categorical',
        'importance': 'High - This is what we predict'
    },
    
    # Pressure Sensors
    'aa_000': {
        'description': 'Compressor outlet pressure - measures air pressure coming out of the compressor',
        'unit': 'bar/psi',
        'typical_range': '8-12 bar',
        'type': 'continuous',
        'importance': 'Critical - Indicates compressor health'
    },
    'ac_000': {
        'description': 'Reservoir 1 pressure - main air tank pressure',
        'unit': 'bar/psi',
        'typical_range': '10-12 bar',
        'type': 'continuous',
        'importance': 'Critical - Main air storage'
    },
    'ad_000': {
        'description': 'Reservoir 2 pressure - secondary air tank pressure',
        'unit': 'bar/psi',
        'typical_range': '10-12 bar',
        'type': 'continuous',
        'importance': 'Critical - Backup air storage'
    },
    'ae_000': {
        'description': 'Brake circuit 1 pressure - pressure in primary brake line',
        'unit': 'bar/psi',
        'typical_range': '8-10 bar',
        'type': 'continuous',
        'importance': 'Critical - Brake performance'
    },
    'af_000': {
        'description': 'Brake circuit 2 pressure - pressure in secondary brake line',
        'unit': 'bar/psi',
        'typical_range': '8-10 bar',
        'type': 'continuous',
        'importance': 'Critical - Brake redundancy'
    },
    
    # Temperature Sensors
    'ag_000': {
        'description': 'Compressor temperature - reading 1',
        'unit': '°C',
        'typical_range': '70-95°C',
        'type': 'continuous',
        'importance': 'High - Indicates compressor overheating'
    },
    'ag_001': {
        'description': 'Compressor temperature - reading 2',
        'unit': '°C',
        'typical_range': '70-95°C',
        'type': 'continuous',
        'importance': 'High - Temperature trend'
    },
    'ag_002': {
        'description': 'Compressor temperature - reading 3',
        'unit': '°C',
        'typical_range': '70-95°C',
        'type': 'continuous',
        'importance': 'Medium - Part of temperature array'
    },
    'ag_003': {
        'description': 'Compressor temperature - reading 4',
        'unit': '°C',
        'typical_range': '70-95°C',
        'type': 'continuous',
        'importance': 'Medium - Part of temperature array'
    },
    'ag_004': {
        'description': 'Compressor temperature - reading 5',
        'unit': '°C',
        'typical_range': '70-95°C',
        'type': 'continuous',
        'importance': 'Medium - Part of temperature array'
    },
    'ag_005': {
        'description': 'Compressor temperature - reading 6',
        'unit': '°C',
        'typical_range': '70-95°C',
        'type': 'continuous',
        'importance': 'Medium - Part of temperature array'
    },
    'ag_006': {
        'description': 'Compressor temperature - reading 7',
        'unit': '°C',
        'typical_range': '70-95°C',
        'type': 'continuous',
        'importance': 'Medium - Part of temperature array'
    },
    'ag_007': {
        'description': 'Compressor temperature - reading 8',
        'unit': '°C',
        'typical_range': '70-95°C',
        'type': 'continuous',
        'importance': 'Medium - Part of temperature array'
    },
    'ag_008': {
        'description': 'Compressor temperature - reading 9',
        'unit': '°C',
        'typical_range': '70-95°C',
        'type': 'continuous',
        'importance': 'Medium - Part of temperature array'
    },
    'ag_009': {
        'description': 'Compressor temperature - reading 10',
        'unit': '°C',
        'typical_range': '70-95°C',
        'type': 'continuous',
        'importance': 'Medium - Part of temperature array'
    },
    'ah_000': {
        'description': 'Engine compartment temperature',
        'unit': '°C',
        'typical_range': '80-110°C',
        'type': 'continuous',
        'importance': 'Medium - Ambient under-hood temperature'
    },
    'ai_000': {
        'description': 'Ambient (outside) temperature',
        'unit': '°C',
        'typical_range': '-30 to 50°C',
        'type': 'continuous',
        'importance': 'Low - Environmental factor'
    },
    
    # System Status Sensors
    'aj_000': {
        'description': 'Engine speed (RPM)',
        'unit': 'RPM',
        'typical_range': '600-2500 RPM',
        'type': 'continuous',
        'importance': 'Medium - Correlates with air demand'
    },
    'ak_000': {
        'description': 'Vehicle speed',
        'unit': 'km/h',
        'typical_range': '0-120 km/h',
        'type': 'continuous',
        'importance': 'Medium - Brake usage frequency'
    },
    'al_000': {
        'description': 'Battery voltage',
        'unit': 'V',
        'typical_range': '12-14.5V',
        'type': 'continuous',
        'importance': 'Low - Electrical system health'
    },
    'am_0': {
        'description': 'System current draw',
        'unit': 'A',
        'typical_range': '50-200A',
        'type': 'continuous',
        'importance': 'Low - Electrical load'
    },
    
    # Valve Position Sensors
    'an_000': {
        'description': 'Pressure regulator valve position',
        'unit': '%',
        'typical_range': '0-100%',
        'type': 'continuous',
        'importance': 'High - Controls system pressure'
    },
    'ao_000': {
        'description': 'Safety valve position',
        'unit': '%',
        'typical_range': '0-100%',
        'type': 'continuous',
        'importance': 'High - Emergency pressure release'
    },
    'ap_000': {
        'description': 'Brake valve position - front axle',
        'unit': '%',
        'typical_range': '0-100%',
        'type': 'continuous',
        'importance': 'High - Front brake application'
    },
    'aq_000': {
        'description': 'Brake valve position - rear axle',
        'unit': '%',
        'typical_range': '0-100%',
        'type': 'continuous',
        'importance': 'High - Rear brake application'
    },
    
    # Pressure Sensor Arrays (Multiple Readings)
    'ay_000': {
        'description': 'Pressure fluctuation reading 1',
        'unit': 'bar',
        'typical_range': '8-12 bar',
        'type': 'continuous',
        'importance': 'High - Dynamic pressure behavior'
    },
    'ay_001': {
        'description': 'Pressure fluctuation reading 2',
        'unit': 'bar',
        'typical_range': '8-12 bar',
        'type': 'continuous',
        'importance': 'High - Pressure stability'
    },
    'ay_002': {
        'description': 'Pressure fluctuation reading 3',
        'unit': 'bar',
        'typical_range': '8-12 bar',
        'type': 'continuous',
        'importance': 'Medium - Part of pressure array'
    },
    'ay_003': {
        'description': 'Pressure fluctuation reading 4',
        'unit': 'bar',
        'typical_range': '8-12 bar',
        'type': 'continuous',
        'importance': 'Medium - Part of pressure array'
    },
    'ay_004': {
        'description': 'Pressure fluctuation reading 5',
        'unit': 'bar',
        'typical_range': '8-12 bar',
        'type': 'continuous',
        'importance': 'Medium - Part of pressure array'
    },
    'ay_005': {
        'description': 'Pressure fluctuation reading 6',
        'unit': 'bar',
        'typical_range': '8-12 bar',
        'type': 'continuous',
        'importance': 'Medium - Part of pressure array'
    },
    'ay_006': {
        'description': 'Pressure fluctuation reading 7',
        'unit': 'bar',
        'typical_range': '8-12 bar',
        'type': 'continuous',
        'importance': 'Medium - Part of pressure array'
    },
    'ay_007': {
        'description': 'Pressure fluctuation reading 8',
        'unit': 'bar',
        'typical_range': '8-12 bar',
        'type': 'continuous',
        'importance': 'Medium - Part of pressure array'
    },
    'ay_008': {
        'description': 'Pressure fluctuation reading 9',
        'unit': 'bar',
        'typical_range': '8-12 bar',
        'type': 'continuous',
        'importance': 'Medium - Part of pressure array'
    },
    'ay_009': {
        'description': 'Pressure fluctuation reading 10',
        'unit': 'bar',
        'typical_range': '8-12 bar',
        'type': 'continuous',
        'importance': 'Medium - Part of pressure array'
    },
    
    # Flow Rate Sensors
    'az_000': {
        'description': 'Air flow rate reading 1',
        'unit': 'L/min',
        'typical_range': '100-500 L/min',
        'type': 'continuous',
        'importance': 'High - System air consumption'
    },
    'az_001': {
        'description': 'Air flow rate reading 2',
        'unit': 'L/min',
        'typical_range': '100-500 L/min',
        'type': 'continuous',
        'importance': 'High - Flow stability'
    }
}

# Add remaining sensor descriptions automatically
for i in range(2, 10):
    for prefix in ['az', 'ba', 'bb', 'bc', 'bd', 'be', 'bf', 'bg', 'bh', 'bi', 'bj', 'bk', 'bl', 'bm']:
        col = f'{prefix}_00{i}'
        if col not in COLUMN_DESCRIPTIONS:
            COLUMN_DESCRIPTIONS[col] = {
                'description': f'{prefix.upper()} sensor reading {i}',
                'unit': 'various',
                'typical_range': 'varies',
                'type': 'continuous',
                'importance': 'Medium'
            }

# Add CN sensors
for i in range(2, 10):
    col = f'cn_00{i}'
    if col not in COLUMN_DESCRIPTIONS:
        COLUMN_DESCRIPTIONS[col] = {
            'description': f'Compressor vibration reading {i+1}',
            'unit': 'mm/s',
            'typical_range': '0-10 mm/s',
            'type': 'continuous',
            'importance': 'Medium'
        }

# Add CS sensors
for i in range(2, 10):
    col = f'cs_00{i}'
    if col not in COLUMN_DESCRIPTIONS:
        COLUMN_DESCRIPTIONS[col] = {
            'description': f'Sensor signal voltage reading {i+1}',
            'unit': 'V',
            'typical_range': '0-5V',
            'type': 'continuous',
            'importance': 'Medium'
        }

# Add EE sensors
for i in range(2, 10):
    col = f'ee_00{i}'
    if col not in COLUMN_DESCRIPTIONS:
        COLUMN_DESCRIPTIONS[col] = {
            'description': f'Actuator response time reading {i+1}',
            'unit': 'ms',
            'typical_range': '50-200 ms',
            'type': 'continuous',
            'importance': 'Medium'
        }

# Dropped columns
DROPPED_COLUMNS = ['br_000', 'bq_000', 'bp_000', 'ab_000', 'cr_000', 'bo_000', 'bn_000']

def show():
    st.title("📋 Project Information")
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview", 
        "🔧 Sensor Columns", 
        "📝 Column Descriptions",
        "🎯 Target Variable", 
        "🤖 Model Performance",
        "📈 Business Impact"
    ])
    
    # ==================== TAB 1: OVERVIEW ====================
    with tab1:
        st.header("APS Sensor Fault Detection System")

        # Load dataset using your specific database and collection
        df = load_dataset()

        if df is not None:
            st.success("✅ Dataset Loaded Successfully")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Rows", f"{df.shape[0]:,}")
            col2.metric("Total Columns", df.shape[1])
            col3.metric("Total Sensors", df.shape[1] - 1 if 'class' in df.columns else df.shape[1])
            
            # Count missing values
            missing_count = df.isna().sum().sum()
            col4.metric("Missing Values", f"{missing_count:,}")

            st.markdown("### Dataset Preview")
            st.dataframe(df.head(), use_container_width=True)

            # Show data types
            with st.expander("📊 Data Types"):
                dtypes_df = pd.DataFrame({
                    'Column': df.dtypes.index,
                    'Data Type': df.dtypes.values.astype(str)
                })
                st.dataframe(dtypes_df, use_container_width=True)

            # Show missing values
            with st.expander("🔍 Missing Values Analysis"):
                missing_df = (
                    df.isna()
                    .sum()
                    .reset_index()
                    .rename(columns={"index": "Column", 0: "Missing Values"})
                    .sort_values(by="Missing Values", ascending=False)
                )
                missing_df['Percentage'] = (missing_df['Missing Values'] / len(df) * 100).round(2)
                missing_df = missing_df[missing_df['Missing Values'] > 0]
                
                if not missing_df.empty:
                    st.dataframe(missing_df, use_container_width=True)
                    
                    # Plot missing values
                    fig = px.bar(
                        missing_df.head(20),
                        x='Column',
                        y='Percentage',
                        title="Top 20 Columns with Missing Values (%)"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.success("✅ No missing values found in dataset!")
        else:
            st.warning("⚠️ No dataset loaded. Please check:")
            st.info(f"""
            Database: **aps_fault_sensor**
            Collection: **sensor**
            
            Please ensure:
            1. MongoDB is running
            2. Database 'aps_fault_sensor' exists
            3. Collection 'sensor' contains data
            """)
    
    # ==================== TAB 2: SENSOR COLUMNS ====================
    with tab2:
        st.header("🔧 Sensor Columns Overview")
        
        st.markdown("""
        The dataset contains **170+ sensor readings** from various components of the Air Pressure System.
        Sensors are grouped by function and location.
        """)
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        df = load_dataset()
        if df is not None:
            total_sensors = len([col for col in df.columns if col != "class"])
            high_importance = len([c for c in COLUMN_DESCRIPTIONS if COLUMN_DESCRIPTIONS[c]['importance'] in ['High', 'Critical']])
            array_sensors = len([c for c in df.columns if "_00" in c and c[-3:] != "000"])

            col1.metric("Total Sensors", total_sensors)
            col2.metric("High Importance", high_importance)
            col3.metric("Array Sensors", array_sensors)
            col4.metric("Dropped Columns", len(DROPPED_COLUMNS))
        else:
            col1.metric("Total Sensors", len(COLUMN_DESCRIPTIONS))
            col2.metric("High Importance", len([c for c in COLUMN_DESCRIPTIONS if COLUMN_DESCRIPTIONS[c]['importance'] in ['High', 'Critical']]))
            col3.metric("Dropped Columns", len(DROPPED_COLUMNS))
            col4.metric("Target Column", "class")
        
        # Sensor groups
        sensor_groups = {
            "🏭 Pressure Sensors": [col for col in COLUMN_DESCRIPTIONS if any(col.startswith(p) for p in ['aa', 'ac', 'ad', 'ae', 'af', 'ay'])],
            "🌡️ Temperature Sensors": [col for col in COLUMN_DESCRIPTIONS if any(col.startswith(p) for p in ['ag', 'ah', 'ai'])],
            "⚙️ Valve Position": [col for col in COLUMN_DESCRIPTIONS if any(col.startswith(p) for p in ['an', 'ao', 'ap', 'aq'])],
            "📊 Flow Rate": [col for col in COLUMN_DESCRIPTIONS if col.startswith('az')],
            "📈 Vibration": [col for col in COLUMN_DESCRIPTIONS if col.startswith('cn')],
            "🔌 Electrical": [col for col in COLUMN_DESCRIPTIONS if any(col.startswith(p) for p in ['al', 'am', 'cs'])],
            "⏱️ Actuator Response": [col for col in COLUMN_DESCRIPTIONS if col.startswith('ee')],
            "📋 Other Sensors": [col for col in COLUMN_DESCRIPTIONS if not any(col.startswith(p) for p in ['aa', 'ac', 'ad', 'ae', 'af', 'ag', 'ah', 'ai', 'an', 'ao', 'ap', 'aq', 'az', 'cn', 'al', 'am', 'cs', 'ee', 'ay'])]
        }
        
        for group_name, sensors in sensor_groups.items():
            if sensors:
                with st.expander(f"{group_name} ({len(sensors)} sensors)"):
                    cols = st.columns(3)
                    for i, sensor in enumerate(sorted(sensors)[:30]):  # Show first 30
                        desc = COLUMN_DESCRIPTIONS.get(sensor, {})
                        imp = desc.get('importance', 'Medium')
                        emoji = "🔴" if imp in ['High', 'Critical'] else "🟡" if imp == 'Medium' else "⚪"
                        cols[i % 3].write(f"{emoji} `{sensor}`")
                    if len(sensors) > 30:
                        st.write(f"... and {len(sensors)-30} more")

        if df is not None:
            st.markdown("### Sensor Statistics")
            numeric_df = df.select_dtypes(include=["int64", "float64"])
            stats_df = numeric_df.describe().T
            st.dataframe(stats_df.head(20), use_container_width=True)
    
    # ==================== TAB 3: DETAILED COLUMN DESCRIPTIONS ====================
    with tab3:
        st.header("📝 Detailed Column Descriptions")
        
        # Search and filter
        search = st.text_input("🔍 Search for sensor", placeholder="Enter column name (e.g., aa_000)")
        
        # Importance filter
        importance_filter = st.multiselect(
            "Filter by importance",
            options=["Critical", "High", "Medium", "Low"],
            default=["Critical", "High", "Medium"]
        )
        
        # Type filter
        type_filter = st.multiselect(
            "Filter by type",
            options=["continuous", "categorical"],
            default=["continuous"]
        )
        
        # Display columns in a nice table
        descriptions_data = []
        for col, desc in COLUMN_DESCRIPTIONS.items():
            if search and search.lower() not in col.lower():
                continue
            if desc.get('importance', 'Medium') not in importance_filter:
                continue
            if desc.get('type', 'continuous') not in type_filter:
                continue
                
            descriptions_data.append({
                'Column': col,
                'Description': desc.get('description', 'N/A'),
                'Unit': desc.get('unit', 'N/A'),
                'Typical Range': desc.get('typical_range', 'N/A'),
                'Importance': desc.get('importance', 'Medium'),
                'Type': desc.get('type', 'continuous')
            })
        
        if descriptions_data:
            df_desc = pd.DataFrame(descriptions_data)
            st.dataframe(df_desc, use_container_width=True, height=500)
            st.caption(f"Showing {len(descriptions_data)} of {len(COLUMN_DESCRIPTIONS)} sensors")
        else:
            st.warning("No sensors match your filters")
        
        # Dropped columns section
        with st.expander("❌ Dropped Columns"):
            st.markdown("These columns were removed during preprocessing:")
            for col in DROPPED_COLUMNS:
                st.write(f"- `{col}`")
    
    # ==================== TAB 4: TARGET VARIABLE ====================
    with tab4:
        st.header("🎯 Target Variable")

        df = load_dataset()

        if df is not None and "class" in df.columns:
            class_counts = df["class"].value_counts()

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Class Summary")
                st.write(f"**Total Samples:** {df.shape[0]:,}")
                st.write(f"**Normal (neg):** {class_counts.get('neg', 0):,}")
                st.write(f"**Failure (pos):** {class_counts.get('pos', 0):,}")

                if class_counts.get('pos', 0) > 0:
                    imbalance_ratio = class_counts.get('neg', 0) / class_counts.get('pos', 1)
                    st.write(f"**Imbalance Ratio (neg/pos):** {imbalance_ratio:.2f}")

            with col2:
                fig = px.pie(
                    values=class_counts.values,
                    names=class_counts.index,
                    title="Actual Class Distribution",
                    color_discrete_sequence=['green', 'red']
                )
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Why Imbalance Matters")
            st.info("""
            APS dataset is typically highly imbalanced.
            - **False Negative** → Truck breakdown (very costly)
            - **False Positive** → Inspection cost (minor)
            """)
        else:
            st.warning("No class column found in dataset")
    
    # ==================== TAB 5: MODEL PERFORMANCE ====================
    with tab5:
        st.header("🤖 Model Performance")
        
        try:
            mongo_client = MongoDBClient()
            db = mongo_client.database
            
            if "predictions" not in db.list_collection_names():
                st.error("❌ 'predictions' collection not found")
            else:
                total = db.predictions.count_documents({})

                if total == 0:
                    st.warning("No prediction data available")
                else:
                    faults = db.predictions.count_documents({"prediction": 1})

                    # Confidence aggregation
                    pipeline = [{
                        "$group": {
                            "_id": None,
                            "avg_conf": {"$avg": "$confidence"}
                        }
                    }]

                    result = list(db.predictions.aggregate(pipeline))
                    avg_conf = result[0]["avg_conf"] if result else 0

                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total Predictions", f"{total:,}")
                    col2.metric("Faults Detected", f"{faults:,}")
                    col3.metric("Fault Rate", f"{(faults/total*100):.2f}%")
                    col4.metric("Avg Confidence", f"{avg_conf:.2%}" if avg_conf else "N/A")

                    # Pie Chart
                    fig = px.pie(
                        values=[total - faults, faults],
                        names=["Normal", "Fault"],
                        title="Prediction Distribution",
                        color_discrete_sequence=['green', 'red']
                    )
                    st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Database Error: {e}")
    
    # ==================== TAB 6: BUSINESS IMPACT ====================
    with tab6:
        st.header("📈 Business Impact & ROI")
        
        try:
            mongo_client = MongoDBClient()
            db = mongo_client.database

            if "predictions" not in db.list_collection_names():
                st.error("No predictions collection found")
            else:
                total = db.predictions.count_documents({})

                if total == 0:
                    st.warning("No predictions available")
                else:
                    faults = db.predictions.count_documents({"prediction": 1})

                    st.subheader("💰 Financial Estimation")

                    col1, col2 = st.columns(2)
                    with col1:
                        cost_per_fault = st.number_input(
                            "Cost per Fault ($)", 1000, 1000000, 50000, step=1000)

                    with col2:
                        saving_per_fault = st.number_input(
                            "Saving per Prevention ($)", 1000, 1000000, 30000, step=1000)

                    total_savings = faults * saving_per_fault
                    prevented_loss = faults * cost_per_fault

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Faults Detected", f"{faults:,}")
                    col2.metric("Potential Savings", f"${total_savings:,.0f}")
                    col3.metric("Loss Prevented", f"${prevented_loss:,.0f}")

                    system_cost = 50000
                    roi = ((total_savings - system_cost) / system_cost) * 100

                    st.metric("Estimated ROI", f"{roi:.2f}%")

        except Exception as e:
            st.error(f"Business Impact Error: {e}")
    
    # Footer
    st.markdown("---")
    st.write(f"APS Sensor Fault Detection System | Built with Streamlit | Scania Truck Dataset")
    st.write(f"📊 {len(COLUMN_DESCRIPTIONS)} sensors monitored | 🎯 Predicting APS failures | 💰 Maximizing fleet uptime")
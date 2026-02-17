import streamlit as st
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from sensor.configuration.mongo_db_connection import MongoDBClient

st.set_page_config(
    page_title="APS Sensor Dashboard",
    page_icon="🔧",
    layout="wide"
)

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = "Live Monitoring"

# SIDEBAR ONLY - NO OTHER CONTENT
with st.sidebar:
    st.title("🔧 APS Sensor Dashboard")
    st.markdown("---")
    
    # Navigation - PREDICT ADDED HERE
    pages = [
        "Live Monitoring", 
        "ROI Analytics", 
        "Active Learning", 
        "Model Insights",
        "🔮 Predict"  # ← ADDED PREDICT WITH EMOJI
    ]
    
    selected = st.radio("Navigation", pages)
    st.session_state.page = selected
    
    st.markdown("---")
    
    # Database status
    try:
        client = MongoDBClient()
        db = client.database
        if 'predictions' in db.list_collection_names():
            count = db.predictions.count_documents({})
            st.success(f"✅ Connected")
            st.caption(f"📊 {count} predictions")
        else:
            st.warning("⚠️ No predictions")
    except Exception as e:
        st.error(f"❌ DB Error")

# MAIN CONTENT AREA - ROUTING WITH PREDICT
try:
    if st.session_state.page == "Live Monitoring":
        from pages.live_monitoring import show
        show()
    elif st.session_state.page == "ROI Analytics":
        from pages.roi_analytics import show
        show()
    elif st.session_state.page == "Active Learning":
        from pages.active_learning import show
        show()
    elif st.session_state.page == "Model Insights":
        from pages.model_insights import show
        show()
    elif st.session_state.page == "🔮 Predict":  # ← ADDED PREDICT ROUTING
        from pages.predict import show
        show()

        
except Exception as e:
    st.error(f"❌ Error loading page: {e}")
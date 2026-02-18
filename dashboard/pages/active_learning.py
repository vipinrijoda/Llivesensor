
import streamlit as st
import pandas as pd
from datetime import datetime
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from sensor.configuration.mongo_db_connection import MongoDBClient

# Your sensor columns
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

def show():
    st.title("🧪 Active Learning") 
    
    st.info("Review uncertain predictions to help improve the model")
    
    try:
        client = MongoDBClient()
        db = client.database
        
        if 'predictions' not in db.list_collection_names():
            st.error("❌ No 'predictions' collection found")
            return
        
        # Get predictions with low confidence that haven't been reviewed
        pending = list(db.predictions.find({
            'confidence': {'$lt': 0.8},
            'reviewed': {'$ne': True}
        }).limit(20))
        
        if not pending:
            st.success("✅ No pending reviews! All predictions have high confidence.")
            return
            
        df = pd.DataFrame(pending)
        
        # Handle MongoDB ObjectId
        if '_id' in df.columns:
            df['_id'] = df['_id'].astype(str)
        
        # Get available sensors
        available_sensors = [col for col in ALL_SENSOR_COLUMNS if col in df.columns]
        
        st.metric("Pending Reviews", len(df))
        
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        col1.metric("Avg Confidence", f"{df['confidence'].mean():.1%}")
        col2.metric("Fault Predictions", int(df[df['prediction']==1].shape[0]) if 'prediction' in df.columns else 0)
        col3.metric("Normal Predictions", int(df[df['prediction']==0].shape[0]) if 'prediction' in df.columns else 0)
        
        # Review interface
        st.subheader("📋 Cases Needing Review")
        
        for idx, row in df.iterrows():
            with st.expander(f"Case {idx+1} - {row.get('timestamp', 'Unknown')} (Confidence: {row.get('confidence', 0):.1%})"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Prediction Details:**")
                    st.write(f"**Timestamp:** {row.get('timestamp', 'N/A')}")
                    st.write(f"**Model Prediction:** {'⚠️ FAULT' if row.get('prediction')==1 else '✅ NORMAL'}")
                    st.write(f"**Confidence:** {row.get('confidence', 0):.1%}")
                    
                    # Show top sensor values
                    if available_sensors:
                        st.write("**Top Sensor Readings:**")
                        for sensor in available_sensors[:5]:
                            if sensor in row:
                                st.write(f"- {sensor}: {row[sensor]:.2f}")
                
                with col2:
                    st.write("**Your Feedback:**")
                    
                    # Create three columns for feedback options
                    fb_col1, fb_col2, fb_col3 = st.columns(3)
                    
                    with fb_col1:
                        if st.button("✅ Correct", key=f"corr_{idx}"):
                            # Update database
                            db.predictions.update_one(
                                {'_id': row['_id']},
                                {'$set': {
                                    'reviewed': True, 
                                    'human_feedback': 'correct',
                                    'reviewed_at': datetime.now()
                                }}
                            )
                            st.success("✓ Feedback recorded - Thank you!")
                            st.rerun()
                    
                    with fb_col2:
                        if st.button("❌ Incorrect", key=f"inc_{idx}"):
                            db.predictions.update_one(
                                {'_id': row['_id']},
                                {'$set': {
                                    'reviewed': True, 
                                    'human_feedback': 'incorrect',
                                    'reviewed_at': datetime.now(),
                                    'correct_prediction': 1 - row.get('prediction', 0)  # Flip the prediction
                                }}
                            )
                            st.warning("✓ Feedback recorded - This will be used for retraining")
                            st.rerun()
                    
                    with fb_col3:
                        if st.button("⏰ Later", key=f"later_{idx}"):
                            st.info("Saved for later")
        
        # Show feedback statistics
        st.subheader("📊 Feedback Statistics")
        
        # Get feedback stats from database
        reviewed = list(db.predictions.find({'reviewed': True}).limit(1000))
        if reviewed:
            review_df = pd.DataFrame(reviewed)
            if 'human_feedback' in review_df.columns:
                feedback_counts = review_df['human_feedback'].value_counts()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Reviewed", len(review_df))
                
                with col2:
                    if 'correct' in feedback_counts:
                        agreement_rate = (feedback_counts.get('correct', 0) / len(review_df)) * 100
                        st.metric("Human-Model Agreement", f"{agreement_rate:.1f}%")
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")


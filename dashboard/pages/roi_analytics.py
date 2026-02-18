import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from sensor.configuration.mongo_db_connection import MongoDBClient

def format_timestamp(ts):
    """Helper function to format timestamp without milliseconds"""
    if ts is None:
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
    st.title("ROI Analytics") 
    
    # ROI Parameters
    with st.expander("⚙️ Parameters", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            cost_per_fault = st.number_input("Cost per Undetected Fault ($)", value=50000, step=1000)
            savings_per_prevention = st.number_input("Savings per Prevented Fault ($)", value=30000, step=1000)
        with col2:
            false_alarm_cost = st.number_input("Cost per False Alarm ($)", value=500, step=100)
            system_cost = st.number_input("Annual System Cost ($)", value=50000, step=1000)
    
    try:
        # Connect to MongoDB
        client = MongoDBClient()
        db = client.database
        
        if 'predictions' not in db.list_collection_names():
            st.warning("⚠️ No 'predictions' collection found")
            return
        
        # Date range selector
        col1, col2 = st.columns(2)
        with col1:
            date_option = st.selectbox(
                "Select Time Range",
                ["Last 7 Days", "Last 30 Days", "Last 90 Days", "Last Year", "Custom"]
            )
        
        # Calculate date range
        end_date = datetime.now()
        if date_option == "Last 7 Days":
            start_date = end_date - timedelta(days=7)
        elif date_option == "Last 30 Days":
            start_date = end_date - timedelta(days=30)
        elif date_option == "Last 90 Days":
            start_date = end_date - timedelta(days=90)
        elif date_option == "Last Year":
            start_date = end_date - timedelta(days=365)
        else:  # Custom
            with col2:
                custom_days = st.number_input("Number of days", min_value=1, max_value=730, value=30)
                start_date = end_date - timedelta(days=custom_days)
        
        # Get data from MongoDB
        data = list(db.predictions.find({
            'timestamp': {'$gte': start_date, '$lte': end_date}
        }).sort('timestamp', -1))
        
        if not data:
            st.warning(f"⚠️ No data found for selected period")
            return
            
        df = pd.DataFrame(data)
        
        # Handle MongoDB ObjectId
        if '_id' in df.columns:
            df['_id'] = df['_id'].astype(str)
        
        # Convert timestamp
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        if 'prediction' not in df.columns:
            st.error("❌ 'prediction' column not found in data")
            return
        
        # Calculate metrics
        total_pred = len(df)
        faults = df['prediction'].sum()
        
        # Calculate period in months
        period_days = (end_date - start_date).days
        period_months = period_days / 30.44  # Average days per month
        
        # Estimate false positives (simplified - assume 95% accuracy)
        accuracy = 0.95
        false_positives = int(total_pred * (1 - accuracy) * 0.5)
        false_negatives = int(total_pred * (1 - accuracy) * 0.5)
        
        # Calculate savings and costs
        prevention_savings = faults * savings_per_prevention
        false_alarm_costs = false_positives * false_alarm_cost
        missed_fault_costs = false_negatives * cost_per_fault
        system_cost_period = (system_cost / 12) * period_months
        
        total_costs = false_alarm_costs + missed_fault_costs + system_cost_period
        net_savings = prevention_savings - total_costs
        roi = (net_savings / system_cost_period) * 100 if system_cost_period > 0 else 0
        
        # Display metrics
        st.subheader(f"📊 ROI Summary ({period_days} days)")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Predictions", f"{total_pred:,}")
        with col2:
            st.metric("Faults Detected", f"{int(faults):,}")
        with col3:
            st.metric("False Positives (est.)", f"{false_positives:,}")
        with col4:
            st.metric("False Negatives (est.)", f"{false_negatives:,}")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Prevention Savings", f"${prevention_savings:,.0f}")
        with col2:
            st.metric("Total Costs", f"${total_costs:,.0f}")
        with col3:
            st.metric("Net Savings", f"${net_savings:,.0f}")
        with col4:
            st.metric("ROI", f"{roi:.1f}%")
        
        # Daily faults chart - WITH FIXED TIME FORMAT
        if 'timestamp' in df.columns:
            st.subheader("📈 Daily Faults Trend")
            
            # Create date column for grouping
            df['date'] = pd.to_datetime(df['timestamp']).dt.date
            daily_faults = df.groupby('date').agg({
                'prediction': ['sum', 'count']
            }).reset_index()
            daily_faults.columns = ['date', 'faults', 'total']
            daily_faults['fault_rate'] = (daily_faults['faults'] / daily_faults['total'] * 100)
            
            # Create figure with dual axis
            fig = go.Figure()
            
            # Bar chart for faults
            fig.add_trace(go.Bar(
                x=daily_faults['date'],
                y=daily_faults['faults'],
                name='Faults',
                marker_color='red',
                yaxis='y'
            ))
            
            # Line chart for fault rate
            fig.add_trace(go.Scatter(
                x=daily_faults['date'],
                y=daily_faults['fault_rate'],
                name='Fault Rate %',
                mode='lines+markers',
                line=dict(color='blue', width=2),
                yaxis='y2'
            ))
            
            fig.update_layout(
                title="Daily Faults and Fault Rate",
                xaxis_title="Date",
                yaxis=dict(title="Number of Faults", side='left'),
                yaxis2=dict(
                    title="Fault Rate (%)",
                    overlaying='y',
                    side='right',
                    range=[0, max(daily_faults['fault_rate'].max() * 1.1, 10)]
                ),
                height=400,
                hovermode='x unified',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            # Format x-axis dates
            fig.update_xaxes(tickformat="%Y-%m-%d")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Cumulative savings chart
            st.subheader("💰 Cumulative Savings")
            
            daily_faults['cumulative_faults'] = daily_faults['faults'].cumsum()
            daily_faults['cumulative_savings'] = daily_faults['cumulative_faults'] * savings_per_prevention
            
            fig = px.line(
                daily_faults,
                x='date',
                y='cumulative_savings',
                title="Cumulative Savings Over Time",
                labels={'date': 'Date', 'cumulative_savings': 'Savings ($)'}
            )
            fig.update_traces(line=dict(color='green', width=3))
            fig.update_layout(height=400)
            fig.update_xaxes(tickformat="%Y-%m-%d")
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Cost breakdown pie chart
        st.subheader("🥧 Cost Breakdown")
        
        fig = go.Figure(data=[go.Pie(
            labels=['Prevention Savings', 'False Alarm Costs', 'Missed Fault Costs', 'System Cost'],
            values=[prevention_savings, false_alarm_costs, missed_fault_costs, system_cost_period],
            hole=0.3,
            marker_colors=['green', 'orange', 'red', 'gray'],
            textinfo='label+percent',
            hoverinfo='label+value+percent'
        )])
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed metrics table - WITH FIXED TIME FORMAT
        with st.expander("📋 Detailed Metrics", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Period Information**")
                st.write(f"**Start Date:** {format_timestamp(start_date)}")
                st.write(f"**End Date:** {format_timestamp(end_date)}")
                st.write(f"**Total Days:** {period_days}")
                st.write(f"**Total Months:** {period_months:.1f}")
            
            with col2:
                st.markdown("**Cost Parameters**")
                st.write(f"**Cost per Fault:** ${cost_per_fault:,.0f}")
                st.write(f"**Savings per Prevention:** ${savings_per_prevention:,.0f}")
                st.write(f"**False Alarm Cost:** ${false_alarm_cost:,.0f}")
                st.write(f"**Monthly System Cost:** ${system_cost/12:,.0f}")
        
        # Data quality check - WITH FIXED TIME FORMAT
        with st.expander("🔍 Data Quality Check", expanded=False):
            st.write("**Data Summary:**")
            st.write(f"**Total Records:** {len(df)}")
            st.write(f"**Unique Days:** {df['date'].nunique() if 'date' in df.columns else 'N/A'}")
            
            if 'confidence' in df.columns:
                st.write(f"**Confidence Range:** {df['confidence'].min():.1%} - {df['confidence'].max():.1%}")
                st.write(f"**Avg Confidence:** {df['confidence'].mean():.1%}")
            
            # Show sample of raw data
            st.write("**Sample Data (first 10 rows):**")
            display_df = df.head(10).copy()
            if 'timestamp' in display_df.columns:
                display_df['timestamp'] = display_df['timestamp'].apply(format_timestamp)
            
            # Select relevant columns
            cols_to_show = ['timestamp', 'prediction']
            if 'confidence' in display_df.columns:
                cols_to_show.append('confidence')
            
            st.dataframe(display_df[cols_to_show], use_container_width=True)
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
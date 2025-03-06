# dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import json
import os
from pathlib import Path
from datetime import datetime, timedelta

from datapipe import ProjectContext

# Set up page configuration
st.set_page_config(
    page_title="Pipeline Monitoring Dashboard",
    page_icon="📊",
    layout="wide"
)

# Initialize project context
project_path = os.environ.get("PROJECT_PATH", "./")
context = ProjectContext(project_path)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["Pipeline Runs", "Model Registry", "Metrics", "System Status"]
)

# Function to get data from SQLite
def get_pipeline_runs():
    conn = sqlite3.connect(str(Path(project_path) / 'tracking' / 'tracker.db'))
    runs_df = pd.read_sql("SELECT * FROM pipeline_runs ORDER BY start_time DESC", conn)
    node_df = pd.read_sql("SELECT * FROM node_runs", conn)
    metrics_df = pd.read_sql("SELECT * FROM metrics", conn)
    conn.close()
    
    # Parse dates
    runs_df['start_time'] = pd.to_datetime(runs_df['start_time'])
    runs_df['end_time'] = pd.to_datetime(runs_df['end_time'])
    
    # Calculate duration
    runs_df['duration'] = (runs_df['end_time'] - runs_df['start_time']).dt.total_seconds()
    
    return runs_df, node_df, metrics_df

def get_models():
    conn = sqlite3.connect(str(Path(project_path) / 'models' / 'models.db'))
    models_df = pd.read_sql("SELECT * FROM models ORDER BY created_at DESC", conn)
    conn.close()
    
    # Parse dates
    models_df['created_at'] = pd.to_datetime(models_df['created_at'])
    
    return models_df

# Pipeline Runs Page
if page == "Pipeline Runs":
    st.title("Pipeline Runs")
    
    # Get data
    runs_df, node_df, _ = get_pipeline_runs()
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Runs", len(runs_df))
    
    with col2:
        success_rate = (runs_df['status'] == 'COMPLETED').mean() * 100
        st.metric("Success Rate", f"{success_rate:.1f}%")
    
    with col3:
        avg_duration = runs_df['duration'].mean()
        st.metric("Avg Duration", f"{avg_duration:.1f}s")
    
    with col4:
        recent_runs = runs_df[runs_df['start_time'] > datetime.now() - timedelta(days=1)]
        st.metric("Runs (24h)", len(recent_runs))
    
    # Pipeline runs table
    st.subheader("Recent Pipeline Runs")
    
    if not runs_df.empty:
        # Format the dataframe for display
        display_df = runs_df.copy()
        display_df['start_time'] = display_df['start_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        display_df['end_time'] = display_df['end_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        display_df['params'] = display_df['params'].apply(lambda x: json.dumps(json.loads(x), indent=2) if x else "")
        
        st.dataframe(
            display_df[['run_id', 'pipeline_name', 'start_time', 'end_time', 'status', 'duration']],
            use_container_width=True
        )
        
        # Pipeline run details
        st.subheader("Pipeline Run Details")
        selected_run = st.selectbox("Select Run", runs_df['run_id'].tolist())
        
        if selected_run:
            run_data = runs_df[runs_df['run_id'] == selected_run].iloc[0]
            run_nodes = node_df[node_df['run_id'] == selected_run]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Pipeline:** {run_data['pipeline_name']}")
                st.write(f"**Status:** {run_data['status']}")
                st.write(f"**Duration:** {run_data['duration']:.2f} seconds")
            
            with col2:
                st.write(f"**Start Time:** {run_data['start_time']}")
                st.write(f"**End Time:** {run_data['end_time']}")
                
                # Parse and display parameters
                if run_data['params']:
                    params = json.loads(run_data['params'])
                    if params:
                        st.write("**Parameters:**")
                        st.json(params)
            
            # Node runs
            st.subheader("Node Runs")
            if not run_nodes.empty:
                run_nodes['start_time'] = pd.to_datetime(run_nodes['start_time'])
                run_nodes['end_time'] = pd.to_datetime(run_nodes['end_time'])
                run_nodes['duration'] = (run_nodes['end_time'] - run_nodes['start_time']).dt.total_seconds()
                
                # Format for display
                display_nodes = run_nodes.copy()
                display_nodes['start_time'] = display_nodes['start_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
                display_nodes['end_time'] = display_nodes['end_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
                
                st.dataframe(
                    display_nodes[['node_name', 'status', 'start_time', 'end_time', 'duration']],
                    use_container_width=True
                )
                
                # Node duration chart
                st.subheader("Node Durations")
                fig, ax = plt.subplots(figsize=(10, 6))
                
                sorted_nodes = run_nodes.sort_values('duration', ascending=False)
                ax.barh(sorted_nodes['node_name'], sorted_nodes['duration'])
                ax.set_xlabel('Duration (seconds)')
                ax.set_ylabel('Node')
                ax.set_title('Node Execution Times')
                
                st.pyplot(fig)
    else:
        st.info("No pipeline runs found.")

# Model Registry Page
elif page == "Model Registry":
    st.title("Model Registry")
    
    # Get models data
    models_df = get_models()
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Models", len(models_df))
    
    with col2:
        unique_models = models_df['name'].nunique()
        st.metric("Model Types", unique_models)
    
    with col3:
        production_models = models_df[models_df['is_production'] == 1]
        st.metric("Production Models", len(production_models))
    
    # Model types filter
    if not models_df.empty:
        model_types = ['All'] + list(models_df['name'].unique())
        selected_type = st.selectbox("Filter by Model Type", model_types)
        
        if selected_type != 'All':
            filtered_models = models_df[models_df['name'] == selected_type]
        else:
            filtered_models = models_df
        
        # Model table
        st.subheader("Models")
        
        # Format for display
        display_models = filtered_models.copy()
        display_models['created_at'] = display_models['created_at'].dt.strftime('%Y-%m-%d %H:%M:%S')
        display_models['is_production'] = display_models['is_production'].map({0: 'No', 1: 'Yes'})
        
        st.dataframe(
            display_models[['model_id', 'name', 'version', 'created_at', 'is_production']],
            use_container_width=True
        )
        
        # Model details
        st.subheader("Model Details")
        selected_model = st.selectbox("Select Model", filtered_models['model_id'].tolist())
        
        if selected_model:
            model_data = filtered_models[filtered_models['model_id'] == selected_model].iloc[0]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Name:** {model_data['name']}")
                st.write(f"**Version:** {model_data['version']}")
                st.write(f"**Created:** {model_data['created_at']}")
                st.write(f"**Production:** {model_data['is_production']}")
            
            with col2:
                st.write(f"**Path:**
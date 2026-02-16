"""
Economic Chart Assistant - Combined Version with Image Support and Preview
Features:
1. Pre-created charts from chart_data_processors with date range filtering
2. LLM-powered customization and summarization using AWS Bedrock with image support
3. General chart creation from uploaded data with preview functionality
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import json
import io
import base64
from datetime import datetime
from chart_generator import ChartGenerator
from llm_handler import LLMHandler
from data_processor import DataProcessor
from chart_data_processors import (
    generate_chart, 
    list_available_charts, 
    get_chart_info,
    CHART_REGISTRY,
    Chart01Processor,
    Chart02Processor,
    Chart03Processor,
    Chart04Processor,
    Chart05Processor,
    Chart06Processor,
    Chart07Processor
)
from chart_database import get_database

# Initialize database
db = get_database()

# Page configuration
st.set_page_config(
    page_title="Economic Chart Assistant - Combined",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if 'charts' not in st.session_state:
    st.session_state.charts = []
if 'data' not in st.session_state:
    st.session_state.data = None
if 'selected_charts' not in st.session_state:
    st.session_state.selected_charts = []
if 'precreated_charts' not in st.session_state:
    st.session_state.precreated_charts = {}
if 'current_chart' not in st.session_state:
    st.session_state.current_chart = None
if 'page' not in st.session_state:
    st.session_state.page = "main"

# Add auto-parsing function
def auto_parse_timeseries(df):
    """Auto-parse and convert data to time series format"""
    # Try to identify date columns
    date_cols = []
    for col in df.columns:
        if any(keyword in col.lower() for keyword in ['date', 'time', 'year', 'month', 'day']):
            try:
                df[col] = pd.to_datetime(df[col], infer_datetime_format=True, errors='coerce').dt.strftime('%Y-%m-%d')
                date_cols.append(col)
            except:
                pass
    
    # If no date columns found, try first column
    if not date_cols and len(df.columns) > 0:
        try:
            df[df.columns[0]] = pd.to_datetime(df[df.columns[0]], infer_datetime_format=True, errors='coerce').dt.strftime('%Y-%m-%d')
            date_cols.append(df.columns[0])
        except:
            pass
    
    # Convert numeric columns
    for col in df.columns:
        if col not in date_cols:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                pass
    
    return df

# Add auto axis range function
def get_auto_axis_ranges(df, x_col, y_cols):
    """Auto-calculate axis ranges based on data"""
    ranges = {}
    
    # X-axis range
    if x_col in df.columns:
        x_min, x_max = df[x_col].min(), df[x_col].max()
        ranges['x_range'] = [x_min, x_max]
    
    # Y-axis range
    if y_cols:
        y_values = []
        for col in y_cols:
            if col in df.columns:
                y_values.extend(df[col].dropna().tolist())
        
        if y_values:
            y_min, y_max = min(y_values), max(y_values)
            # Add 5% padding
            padding = (y_max - y_min) * 0.05
            ranges['y_range'] = [y_min - padding, y_max + padding]
    
    return ranges

# Initialize handlers
@st.cache_resource
def get_llm_handler():
    return LLMHandler()

@st.cache_resource
def get_chart_generator():
    return ChartGenerator()

llm_handler = get_llm_handler()
chart_generator = get_chart_generator()
data_processor = DataProcessor()

# Sidebar
st.sidebar.title("📊 Economic Chart Assistant")
st.sidebar.markdown("---")

# Data Analysis section
st.sidebar.markdown("### 📊 Data Analysis")

if st.sidebar.button("📁 Upload Data & Create Chart", key="nav_upload"):
    st.session_state.page = "upload"
    st.rerun()

if st.sidebar.button("🖼️ Upload Image & Data", key="nav_image_upload"):
    st.session_state.page = "image_upload"
    st.rerun()

st.sidebar.markdown("---")

# Main content
st.title("📈 Economic Research Chart Assistant")

# Determine which page to show
page = st.session_state.page

if page == "upload":
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📁 Upload Your Data")
        uploaded_file = st.file_uploader(
            "Upload CSV or Excel file",
            type=['csv', 'xlsx', 'xls'],
            key="upload_nav"
        )
        
        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                # Auto-parse and convert to time series
                df = auto_parse_timeseries(df)
                st.session_state.data = df
                
                st.success(f"✅ Data loaded: {len(df)} rows, {len(df.columns)} columns")
                st.dataframe(df.head(10), use_container_width=True)
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
        
        if st.button("📊 Use Sample Data", key="sample_nav"):
            dates = pd.date_range('2020-01-01', periods=48, freq='MS')
            st.session_state.data = pd.DataFrame({
                'date': dates.strftime('%Y-%m-%d'),
                'gdp_growth': np.random.normal(2.5, 1.5, 48),
                'inflation': np.random.normal(3.0, 1.0, 48)
            })
            st.success("✅ Sample data loaded!")
            st.rerun()
        
        # Show create chart section after data is loaded
        if st.session_state.data is not None:
            st.markdown("---")
            st.subheader("🎨 Create Chart")
            
            # Column selection
            df = st.session_state.data
            
            # Column selection for chart
            col1_sel, col2_sel = st.columns(2)
            
            with col1_sel:
                # Only show date/time columns for X-axis
                date_cols = []
                for col in df.columns:
                    if any(keyword in col.lower() for keyword in ['date', 'time', 'year', 'month', 'day']):
                        date_cols.append(col)
                
                if not date_cols:
                    date_cols = [df.columns[0]]  # Fallback to first column
                
                x_column = st.selectbox(
                    "X-axis Column (Time/Date)",
                    date_cols,
                    key="x_col_select"
                )
                
                # Show date range and selectors
                if x_column in df.columns:
                    try:
                        # Convert to datetime for range calculation
                        date_series = pd.to_datetime(df[x_column], infer_datetime_format=True, errors='coerce')
                        min_date = date_series.min().date()
                        max_date = date_series.max().date()
                        
                        st.caption(f"Available: {min_date} to {max_date}")
                        
                        start_date = st.date_input(
                            "Start Date",
                            value=min_date,
                            min_value=min_date,
                            max_value=max_date,
                            key="start_date_filter"
                        )
                        
                        end_date = st.date_input(
                            "End Date",
                            value=max_date,
                            min_value=min_date,
                            max_value=max_date,
                            key="end_date_filter"
                        )
                    except:
                        start_date = None
                        end_date = None
            
            with col2_sel:
                # Show numeric columns for Y-axis
                numeric_cols = []
                for col in df.columns:
                    if col != x_column and df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                        numeric_cols.append(col)
                
                y_columns = st.multiselect(
                    "Y-axis Columns (Numeric)",
                    numeric_cols,
                    key="y_col_select"
                )
                
                # Chart type selection
                chart_type = st.selectbox(
                    "Chart Type",
                    ["line", "bar", "scatter", "area"],
                    key="chart_type_select"
                )
            
            user_prompt = st.text_area(
                "Chart Title (optional)",
                placeholder="Enter custom chart title",
                key="create_upload"
            )
            
            if st.button("🚀 Generate Chart", key="gen_upload"):
                if y_columns:
                    with st.spinner("Creating chart..."):
                        try:
                            # Filter data by date range if dates selected
                            filtered_df = st.session_state.data.copy()
                            if start_date and end_date:
                                try:
                                    date_series = pd.to_datetime(filtered_df[x_column], infer_datetime_format=True, errors='coerce')
                                    mask = (date_series.dt.date >= start_date) & (date_series.dt.date <= end_date)
                                    filtered_df = filtered_df[mask]
                                except:
                                    pass
                            
                            # Create chart config with auto ranges
                            chart_config = {
                                "chart_type": chart_type,
                                "x_column": x_column,
                                "y_columns": y_columns,
                                "title": user_prompt if user_prompt else f"{', '.join(y_columns)} over {x_column}",
                                "x_label": x_column,
                                "y_label": ", ".join(y_columns),
                                "show_legend": len(y_columns) > 1
                            }
                            
                            # Add auto axis ranges
                            ranges = get_auto_axis_ranges(filtered_df, x_column, y_columns)
                            chart_config.update(ranges)
                            
                            fig = chart_generator.create_chart(filtered_df, chart_config)
                            
                            # Add legend positioning
                            if chart_config.get("show_legend", False):
                                fig.update_layout(
                                    legend=dict(
                                        orientation="h",
                                        yanchor="top",
                                        y=-0.15,
                                        xanchor="center",
                                        x=0.5
                                    )
                                )
                            
                            # Save chart
                            chart_id = db.save_chart(
                                chart_name=chart_config["title"][:50],
                                chart_type="custom",
                                figure_json=fig.to_json(),
                                chart_config=chart_config,
                                user_prompt=user_prompt or f"Chart: {', '.join(y_columns)} vs {x_column}"
                            )
                            
                            # Save data
                            db.save_chart_data(chart_id, 'chart_data', filtered_df)
                            
                            st.session_state.current_upload_chart = {
                                'id': chart_id,
                                'figure': fig,
                                'config': chart_config,
                                'prompt': user_prompt or f"Chart: {', '.join(y_columns)} vs {x_column}",
                                'data': filtered_df
                            }
                            
                            st.success(f"✅ Chart created and saved! (ID: {chart_id})")
                            st.plotly_chart(fig, use_container_width=True)
                            
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                else:
                    st.warning("Please select at least one Y-axis column")
    
    with col2:
        if st.session_state.data is not None:
            st.subheader("🤖 AI Data Analysis")
            
            if st.button("📊 Data Summary", key="data_summary"):
                with st.spinner("Analyzing data..."):
                    try:
                        df = st.session_state.data
                        st.markdown("### 📊 Data Overview")
                        
                        # Basic info
                        summary = f"""- **Rows**: {len(df)}
- **Columns**: {len(df.columns)}
- **Numeric Columns**: {len(df.select_dtypes(include='number').columns)}
"""
                        st.markdown(summary)
                        
                        # Column descriptions
                        st.markdown("### 📋 Column Details")
                        for col in df.columns:
                            dtype = str(df[col].dtype)
                            sample_val = str(df[col].iloc[0]) if len(df) > 0 else "N/A"
                            st.markdown(f"- **{col}** ({dtype}): Sample value = {sample_val}")
                        
                    except Exception as e:
                        st.error(f"Analysis failed: {str(e)}")
        
        # Show AI options after chart is created
        if 'current_upload_chart' in st.session_state:
            st.markdown("---")
            st.subheader("🤖 AI Wizard")
            
            # Show saved chart info
            chart = st.session_state.current_upload_chart
            
            # Initialize preview state for upload section
            if 'upload_preview_figure' not in st.session_state:
                st.session_state.upload_preview_figure = None
            if 'upload_preview_config' not in st.session_state:
                st.session_state.upload_preview_config = None
            
            # Customization with Preview
            st.markdown("### 🎨 Chart Customization")
            custom_prompt = st.text_area(
                "Describe changes (preview before applying)",
                placeholder="e.g., Change title to 'Recent Inflation Trends', use blue colors, render averages for GDP and inflation",
                key="custom_upload"
            )
            
            col1_preview, col2_preview = st.columns(2)
            
            with col1_preview:
                if st.button("🔍 Preview Changes", key="preview_upload"):
                    if custom_prompt:
                        with st.spinner("Generating preview..."):
                            try:
                                chart = st.session_state.current_upload_chart
                                chart_data = chart.get('data')
                                
                                updated_config = llm_handler.interpret_edit_request(
                                    custom_prompt,
                                    chart['config'],
                                    "AWS Bedrock",
                                    chart_data
                                )
                                
                                # Create preview figure
                                preview_fig = chart_generator.create_chart(chart_data, updated_config)
                                
                                # Handle adding averages
                                if chart_data is not None and ('avg' in custom_prompt.lower() or 'average' in custom_prompt.lower()):
                                    numeric_cols = chart_data.select_dtypes(include='number').columns.tolist()
                                    for col in numeric_cols:
                                        if col in ['gdp_growth', 'inflation'] or any(term in col.lower() for term in ['gdp', 'inflation']):
                                            avg_val = chart_data[col].mean()
                                            preview_fig.add_hline(
                                                y=avg_val,
                                                line_dash="dash",
                                                line_color="red" if 'inflation' in col.lower() else "blue",
                                                annotation_text=f"{col} avg: {avg_val:.2f}",
                                                annotation_position="top right"
                                            )
                                
                                # Store preview
                                st.session_state.upload_preview_figure = preview_fig
                                st.session_state.upload_preview_config = updated_config
                                
                                st.success("✅ Preview generated! Review changes below.")
                                st.rerun()
                                
                            except Exception as e:
                                st.error(f"Preview failed: {str(e)[:100]}...")
            
            with col2_preview:
                if st.button("💾 Apply & Save Changes", key="apply_upload", disabled=st.session_state.upload_preview_figure is None):
                    if st.session_state.upload_preview_figure is not None:
                        with st.spinner("Saving changes..."):
                            try:
                                chart = st.session_state.current_upload_chart
                                
                                # Update saved chart
                                db.update_chart(
                                    chart['id'],
                                    chart_config=st.session_state.upload_preview_config,
                                    figure_json=st.session_state.upload_preview_figure.to_json()
                                )
                                
                                # Update session state
                                st.session_state.current_upload_chart['figure'] = st.session_state.upload_preview_figure
                                st.session_state.current_upload_chart['config'] = st.session_state.upload_preview_config
                                
                                # Clear preview
                                st.session_state.upload_preview_figure = None
                                st.session_state.upload_preview_config = None
                                
                                st.success("✅ Changes applied and saved!")
                                st.rerun()
                                
                            except Exception as e:
                                st.error(f"Save failed: {str(e)[:100]}...")
            
            # Show preview if available
            if st.session_state.upload_preview_figure is not None:
                st.markdown("#### 🔍 Preview")
                st.info("👆 Review the changes above. Click 'Apply & Save Changes' to make them permanent.")
                st.plotly_chart(st.session_state.upload_preview_figure, use_container_width=True, key="upload_preview_chart")
                
                if st.button("🔄 Reset Preview", key="reset_upload_preview"):
                    st.session_state.upload_preview_figure = None
                    st.session_state.upload_preview_config = None
                    st.rerun()
            
            st.markdown("---")
            
            # Quick Analysis
            if st.button("📊 Quick Analysis", key="quick_analysis_upload"):
                with st.spinner("Analyzing chart..."):
                    try:
                        chart = st.session_state.current_upload_chart
                        chart_name = chart.get('config', {}).get('title', chart['prompt'])
                        
                        analysis_prompt = f"""
Analyze this economic chart: {chart_name}

Provide 3 key insights about this chart focusing on:
1. Main trends visible
2. Economic implications
3. Notable patterns

Keep response to 150-200 words.
"""
                        
                        analysis = llm_handler._call_llm(
                            "You are an economic analyst providing chart insights.",
                            analysis_prompt,
                            "AWS Bedrock"
                        )
                        
                        st.markdown("### 🔍 Analysis")
                        st.markdown(analysis)
                        
                    except Exception as e:
                        st.markdown("### 🔍 Analysis")
                        st.markdown(f"**Chart**: {st.session_state.current_upload_chart['prompt']}\n\n**Insight**: This economic visualization shows data patterns that indicate trends worth monitoring for policy and investment decisions.")
                        st.error(f"AI analysis unavailable: {str(e)[:50]}...")

else:
    st.info("👈 Click 'Upload Data & Create Chart' in the sidebar to get started")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info("""
**Economic Chart Assistant**

Features:
- Custom chart creation with preview
- AI-powered customization
- Chart analysis and insights
- Data upload and processing

Using AWS Bedrock for AI
""")
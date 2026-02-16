"""
Economic Chart Assistant - Combined Version with Image Support
Features:
1. Pre-created charts from chart_data_processors with date range filtering
2. LLM-powered customization and summarization using AWS Bedrock with image support
3. General chart creation from uploaded data
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
from metadata_handler import MetadataHandler
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
    page_title="Economic Chart Assistant - Enhanced",
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
metadata_handler = MetadataHandler()

# Add image support methods to LLM handler
def generate_chart_image_context(figure, chart_config, prompt, data):
    """Generate comprehensive chart context with image"""
    try:
        # Generate chart image
        img_bytes = figure.to_image(format="png", width=800, height=600)
        img_base64 = base64.b64encode(img_bytes).decode()
        
        # Extract chart data
        chart_data = []
        for trace in figure.data:
            if hasattr(trace, 'y') and trace.y is not None and len(trace.y) > 0:
                y_values = list(trace.y)
                x_values = list(trace.x) if hasattr(trace, 'x') and trace.x is not None else list(range(len(y_values)))
                chart_data.append({
                    'name': getattr(trace, 'name', 'Data'),
                    'x_values': x_values,
                    'y_values': y_values,
                    'min': min(y_values),
                    'max': max(y_values),
                    'avg': sum(y_values) / len(y_values),
                    'trend': 'increasing' if y_values[-1] > y_values[0] else 'decreasing',
                    'total_points': len(y_values)
                })
        
        return {
            'image_base64': img_base64,
            'chart_data': chart_data,
            'config': chart_config,
            'prompt': prompt,
            'raw_data': data.to_string() if data is not None else 'No raw data'
        }
    except Exception as e:
        return None

# Sidebar
st.sidebar.title("📊 Chart Assistant")

if st.sidebar.button("📁 Upload Data", key="nav_upload", use_container_width=True):
    st.session_state.page = "upload"
    st.rerun()

if st.sidebar.button("🖼️ Image Analysis", key="nav_image_upload", use_container_width=True):
    st.session_state.page = "image_upload"
    st.rerun()

# Saved Charts section
st.sidebar.markdown("---")
if st.sidebar.button("💾 Saved Charts", key="nav_saved", use_container_width=True):
    st.session_state.page = "saved"
    st.rerun()

st.sidebar.markdown("---")
llm_provider = st.sidebar.selectbox("LLM", ["AWS Bedrock", "OpenAI", "Anthropic"], label_visibility="collapsed")

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
        
        # Metadata/Stylesheet upload
        st.markdown("---")
        st.markdown("### 🎨 Upload Metadata/Stylesheet (Optional)")
        metadata_file = st.file_uploader(
            "Upload chart metadata (JSON/YAML)",
            type=['json', 'yaml', 'yml'],
            key="metadata_upload",
            help="Upload a JSON or YAML file with chart styling and configuration"
        )
        
        if metadata_file:
            try:
                file_content = metadata_file.read().decode('utf-8')
                file_type = metadata_file.name.split('.')[-1]
                metadata = metadata_handler.parse_metadata(file_content, file_type)
                st.session_state.chart_metadata = metadata
                st.success("✅ Metadata loaded successfully!")
                with st.expander("📋 View Metadata"):
                    st.json(metadata)
            except Exception as e:
                st.error(f"Metadata error: {str(e)}")
                st.session_state.chart_metadata = None
        
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
            
            # AI-Powered Interactive Chart Builder
            st.markdown("### 🤖 AI Chart Builder")
            ai_prompt = st.text_area(
                "Describe the chart you want to create",
                placeholder="e.g., Show GDP growth and inflation trends over time with blue and red colors",
                key="ai_chart_prompt",
                help="AI will analyze your data, metadata, and reference image to build the chart"
            )
            
            if st.button("✨ Build Chart with AI", key="ai_build_chart"):
                if ai_prompt:
                    with st.spinner("🤖 AI is building your chart..."):
                        try:
                            metadata = st.session_state.get('chart_metadata', None)
                            
                            # Preprocess data - handle long format
                            data_to_use = st.session_state.data.copy()
                            
                            # Extract component info from metadata if available
                            component_info = ""
                            if metadata and 'data_configuration' in metadata and 'components' in metadata['data_configuration']:
                                components = metadata['data_configuration']['components']
                                component_names = [c.get('name', '') for c in components]
                                component_info = f"\nComponents in 'variable' column: {', '.join(component_names)}"
                            
                            # Detect long format (date, variable, value)
                            if 'variable' in data_to_use.columns and 'value' in data_to_use.columns and 'date' in data_to_use.columns:
                                st.info(f"📊 Detected long-format data with components in 'variable' column. Pivoting for visualization...{component_info}")
                                try:
                                    # Remove duplicates by taking mean of duplicate date-variable pairs
                                    data_to_use = data_to_use.groupby(['date', 'variable'], as_index=False)['value'].mean()
                                    # Pivot to wide format
                                    data_to_use = data_to_use.pivot(index='date', columns='variable', values='value').reset_index()
                                    # Flatten column names
                                    data_to_use.columns.name = None
                                except Exception as pivot_error:
                                    st.warning(f"Pivot failed: {str(pivot_error)}. Using original data.")
                                    # Keep original data if pivot fails
                                    pass
                            
                            # Use AI to build chart interactively
                            chart_config = llm_handler.build_chart_interactively(
                                ai_prompt,
                                data_to_use,
                                llm_provider,
                                metadata
                            )
                            
                            # Merge with metadata if provided
                            if metadata:
                                chart_config = metadata_handler.merge_with_config(chart_config, metadata)
                            
                            # Validate config has required fields
                            if 'x_column' not in chart_config or 'y_columns' not in chart_config:
                                raise ValueError("Chart config missing required fields")
                            
                            fig = chart_generator.create_chart(data_to_use, chart_config)
                            
                            # Save chart
                            chart_id = db.save_chart(
                                chart_name=chart_config.get("title", "AI Generated Chart")[:50],
                                chart_type="ai_generated",
                                figure_json=fig.to_json(),
                                chart_config=chart_config,
                                user_prompt=ai_prompt
                            )
                            
                            db.save_chart_data(chart_id, 'chart_data', data_to_use)
                            
                            st.session_state.current_upload_chart = {
                                'id': chart_id,
                                'figure': fig,
                                'config': chart_config,
                                'prompt': ai_prompt,
                                'data': data_to_use
                            }
                            
                            st.success(f"✅ AI-generated chart created! (ID: {chart_id})")
                            
                            # Show generated config for debugging
                            with st.expander("🔧 View Generated Config"):
                                st.json(chart_config)
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                        except Exception as e:
                            st.error(f"AI chart generation failed: {str(e)}")
                            
                            # Show detailed error in expander
                            with st.expander("🔍 View Error Details"):
                                import traceback
                                st.code(traceback.format_exc())
                            
                            # Offer quick manual fallback
                            st.warning("💡 Quick Fix: Try manual chart creation below, or simplify your prompt")
                            
                            # Auto-suggest based on data
                            data_cols = st.session_state.data.columns.tolist()
                            if 'date' in data_cols and 'value' in data_cols:
                                st.info(f"📊 Detected columns: {', '.join(data_cols)}. Try: 'Show value over time by variable'")
                else:
                    st.warning("Please describe the chart you want to create")
            
            st.markdown("---")
            st.markdown("### 🔧 Manual Chart Creation")
            
            # FRED-style customization tabs
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Data", "🎨 Appearance", "📏 Axes", "🔧 Advanced"])
            
            # Column selection
            df = st.session_state.data
            
            with tab1:
                # Column selection for chart
                col1_sel, col2_sel = st.columns(2)
                
                with col1_sel:
                    # Check if data has series identifier column (like 'variable')
                    has_series_col = any(col.lower() in ['variable', 'series', 'category', 'component'] for col in df.columns)
                    
                    if has_series_col:
                        st.markdown("**📊 Data Format Detected: Long Format**")
                        series_col = st.selectbox(
                            "Series Identifier Column",
                            [col for col in df.columns if col.lower() in ['variable', 'series', 'category', 'component']],
                            key="series_col_select",
                            help="Column containing series/component names"
                        )
                        
                        value_col = st.selectbox(
                            "Value Column",
                            [col for col in df.columns if df[col].dtype in ['int64', 'float64', 'int32', 'float32']],
                            key="value_col_select",
                            help="Column containing numeric values"
                        )
                    else:
                        series_col = None
                        value_col = None
                    
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
                        ["line", "bar", "scatter", "area", "stacked_bar", "stacked_area"],
                        key="chart_type_select"
                    )
                
                # Date range presets (outside nested columns)
                st.markdown("**📅 Quick Date Ranges**")
                preset_cols = st.columns(4)
                with preset_cols[0]:
                    if st.button("1Y", key="preset_1y"):
                        end_date = max_date
                        start_date = (pd.Timestamp(max_date) - pd.DateOffset(years=1)).date()
                        st.rerun()
                with preset_cols[1]:
                    if st.button("5Y", key="preset_5y"):
                        end_date = max_date
                        start_date = (pd.Timestamp(max_date) - pd.DateOffset(years=5)).date()
                        st.rerun()
                with preset_cols[2]:
                    if st.button("10Y", key="preset_10y"):
                        end_date = max_date
                        start_date = (pd.Timestamp(max_date) - pd.DateOffset(years=10)).date()
                        st.rerun()
                with preset_cols[3]:
                    if st.button("Max", key="preset_max"):
                        start_date = min_date
                        end_date = max_date
                        st.rerun()
            
            with tab2:
                st.markdown("### 🎨 Chart Appearance")
                
                # Line style options
                col1_app, col2_app = st.columns(2)
                with col1_app:
                    line_width = st.slider("Line Width", 1, 5, 2, key="line_width")
                    line_dash = st.selectbox("Line Style", ["solid", "dash", "dot", "dashdot"], key="line_dash")
                    show_markers = st.checkbox("Show Markers", value=False, key="show_markers")
                    if show_markers:
                        marker_size = st.slider("Marker Size", 4, 12, 6, key="marker_size")
                
                with col2_app:
                    # Color selection for each component
                    if y_columns:
                        st.markdown("**Component Colors**")
                        component_colors = {}
                        for idx, y_col in enumerate(y_columns):
                            component_colors[y_col] = st.color_picker(
                                f"{y_col}",
                                value=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"][idx % 6],
                                key=f"color_{y_col}"
                            )
                
                # Chart dimensions
                st.markdown("**📐 Chart Dimensions**")
                dim_col1, dim_col2 = st.columns(2)
                with dim_col1:
                    chart_width = st.number_input("Width (px)", 600, 1600, 1000, 50, key="chart_width")
                with dim_col2:
                    chart_height = st.number_input("Height (px)", 400, 1200, 600, 50, key="chart_height")
                
                # Legend options
                st.markdown("**📋 Legend**")
                show_legend = st.checkbox("Show Legend", value=len(y_columns) > 1 if y_columns else True, key="show_legend_check")
                if show_legend:
                    legend_position = st.selectbox(
                        "Legend Position",
                        ["top", "bottom", "left", "right", "top-left", "top-right", "bottom-left", "bottom-right"],
                        index=1,
                        key="legend_pos"
                    )
            
            with tab3:
                st.markdown("### 📏 Axis Configuration")
                
                # X-axis options
                st.markdown("**X-Axis (Time)**")
                x_label_custom = st.text_input("X-Axis Label", value=x_column if x_column else "Date", key="x_label")
                x_grid = st.checkbox("Show X Grid", value=True, key="x_grid")
                
                # Y-axis options
                st.markdown("**Y-Axis**")
                y_label_custom = st.text_input("Y-Axis Label", value="Value", key="y_label")
                y_scale = st.selectbox("Y-Axis Scale", ["Linear", "Logarithmic"], key="y_scale")
                y_grid = st.checkbox("Show Y Grid", value=True, key="y_grid")
                
                # Y-axis range
                auto_range = st.checkbox("Auto Y-Axis Range", value=True, key="auto_y_range")
                if not auto_range:
                    y_min = st.number_input("Y-Axis Min", value=0.0, key="y_min")
                    y_max = st.number_input("Y-Axis Max", value=100.0, key="y_max")
            
            with tab4:
                st.markdown("### 🔧 Advanced Options")
                
                # Data transformations
                st.markdown("**📊 Data Transformation**")
                transform = st.selectbox(
                    "Transform",
                    ["None", "Percent Change", "Difference", "Percent Change from Year Ago", "Compounded Annual Rate"],
                    key="data_transform"
                )
                
                # Aggregation
                st.markdown("**📈 Aggregation**")
                aggregation = st.selectbox(
                    "Aggregation Method",
                    ["None", "Average", "Sum", "End of Period"],
                    key="aggregation"
                )
                
                # Annotations
                st.markdown("**📝 Annotations**")
                add_recession_bars = st.checkbox("Add Recession Shading", value=False, key="recession_bars")
                add_reference_line = st.checkbox("Add Reference Line", value=False, key="ref_line")
                if add_reference_line:
                    ref_line_value = st.number_input("Reference Line Value", value=0.0, key="ref_line_val")
                    ref_line_label = st.text_input("Reference Line Label", value="Reference", key="ref_line_label")
            
            # Chart title and subtitle
            st.markdown("---")
            col_title1, col_title2 = st.columns([3, 1])
            with col_title1:
                user_prompt = st.text_input(
                    "Chart Title",
                    placeholder="Enter custom chart title",
                    key="create_upload"
                )
                chart_subtitle = st.text_input(
                    "Subtitle (optional)",
                    placeholder="Additional context or source",
                    key="chart_subtitle"
                )
            
            if st.button("🚀 Generate Chart", key="gen_upload"):
                if y_columns or (series_col and value_col):
                    with st.spinner("Creating chart..."):
                        try:
                            metadata = st.session_state.get('chart_metadata', None)
                            
                            # Filter data by date range if dates selected
                            filtered_df = st.session_state.data.copy()
                            
                            # Convert long format to wide if series column selected
                            if series_col and value_col:
                                st.info(f"🔄 Converting long format to wide format using '{series_col}' as series identifier...")
                                try:
                                    # Aggregate duplicates
                                    filtered_df = filtered_df.groupby([x_column, series_col], as_index=False)[value_col].mean()
                                    # Pivot
                                    filtered_df = filtered_df.pivot(index=x_column, columns=series_col, values=value_col).reset_index()
                                    filtered_df.columns.name = None
                                    # Update y_columns to be the pivoted series
                                    y_columns = [col for col in filtered_df.columns if col != x_column]
                                    st.success(f"✅ Converted to wide format with {len(y_columns)} series")
                                except Exception as e:
                                    st.error(f"Pivot failed: {str(e)}")
                                    st.stop()
                            
                            if start_date and end_date:
                                try:
                                    date_series = pd.to_datetime(filtered_df[x_column], infer_datetime_format=True, errors='coerce')
                                    mask = (date_series.dt.date >= start_date) & (date_series.dt.date <= end_date)
                                    filtered_df = filtered_df[mask]
                                except:
                                    pass
                            
                            # Apply data transformations
                            transform = st.session_state.get('data_transform', 'None')
                            if transform != 'None' and y_columns:
                                for col in y_columns:
                                    if col in filtered_df.columns:
                                        if transform == 'Percent Change':
                                            filtered_df[col] = filtered_df[col].pct_change() * 100
                                        elif transform == 'Difference':
                                            filtered_df[col] = filtered_df[col].diff()
                                        elif transform == 'Percent Change from Year Ago':
                                            filtered_df[col] = filtered_df[col].pct_change(periods=12) * 100
                                        elif transform == 'Compounded Annual Rate':
                                            filtered_df[col] = ((1 + filtered_df[col].pct_change()) ** 12 - 1) * 100
                            
                            # Create chart config with all customizations
                            chart_config = {
                                "chart_type": chart_type,
                                "x_column": x_column,
                                "y_columns": y_columns,
                                "title": user_prompt if user_prompt else f"{', '.join(y_columns)} over {x_column}",
                                "subtitle": st.session_state.get('chart_subtitle', ''),
                                "x_label": st.session_state.get('x_label', x_column),
                                "y_label": st.session_state.get('y_label', ', '.join(y_columns)),
                                "show_legend": st.session_state.get('show_legend_check', len(y_columns) > 1),
                                "legend_position": st.session_state.get('legend_pos', 'bottom'),
                                "line_width": st.session_state.get('line_width', 2),
                                "line_dash": st.session_state.get('line_dash', 'solid'),
                                "show_markers": st.session_state.get('show_markers', False),
                                "marker_size": st.session_state.get('marker_size', 6),
                                "x_grid": st.session_state.get('x_grid', True),
                                "y_grid": st.session_state.get('y_grid', True),
                                "y_scale": st.session_state.get('y_scale', 'Linear'),
                                "chart_width": st.session_state.get('chart_width', 1000),
                                "chart_height": st.session_state.get('chart_height', 600),
                                "transform": transform,
                                "aggregation": st.session_state.get('aggregation', 'None')
                            }
                            
                            # Add component colors
                            if y_columns and 'component_colors' in locals():
                                colors = [component_colors.get(col, "#1f77b4") for col in y_columns]
                                chart_config['colors'] = colors
                            
                            # Add Y-axis range if not auto
                            if not st.session_state.get('auto_y_range', True):
                                chart_config['y_range'] = [st.session_state.get('y_min', 0), st.session_state.get('y_max', 100)]
                            
                            # Merge with metadata if provided
                            if metadata:
                                chart_config = metadata_handler.merge_with_config(chart_config, metadata)
                            
                            # Add auto axis ranges
                            ranges = get_auto_axis_ranges(filtered_df, x_column, y_columns)
                            chart_config.update(ranges)
                            
                            fig = chart_generator.create_chart(filtered_df, chart_config)
                            
                            # Apply advanced customizations
                            legend_pos = chart_config.get('legend_position', 'bottom')
                            legend_config = {
                                'top': dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                                'bottom': dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
                                'left': dict(orientation="v", yanchor="middle", y=0.5, xanchor="right", x=-0.05),
                                'right': dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.02),
                                'top-left': dict(orientation="v", yanchor="top", y=0.99, xanchor="left", x=0.01),
                                'top-right': dict(orientation="v", yanchor="top", y=0.99, xanchor="right", x=0.99),
                                'bottom-left': dict(orientation="v", yanchor="bottom", y=0.01, xanchor="left", x=0.01),
                                'bottom-right': dict(orientation="v", yanchor="bottom", y=0.01, xanchor="right", x=0.99)
                            }
                            
                            if chart_config.get("show_legend", False):
                                fig.update_layout(legend=legend_config.get(legend_pos, legend_config['bottom']))
                            else:
                                fig.update_layout(showlegend=False)
                            
                            # Apply line styles
                            for trace in fig.data:
                                if hasattr(trace, 'line'):
                                    trace.update(
                                        line=dict(
                                            width=chart_config.get('line_width', 2),
                                            dash=chart_config.get('line_dash', 'solid')
                                        )
                                    )
                                if chart_config.get('show_markers', False) and hasattr(trace, 'marker'):
                                    trace.update(
                                        mode='lines+markers',
                                        marker=dict(size=chart_config.get('marker_size', 6))
                                    )
                            
                            # Apply grid settings
                            fig.update_xaxes(showgrid=chart_config.get('x_grid', True))
                            fig.update_yaxes(
                                showgrid=chart_config.get('y_grid', True),
                                type='log' if chart_config.get('y_scale') == 'Logarithmic' else 'linear'
                            )
                            
                            # Add subtitle if provided
                            if chart_config.get('subtitle'):
                                fig.update_layout(
                                    title=dict(
                                        text=f"<b>{chart_config['title']}</b><br><sub>{chart_config['subtitle']}</sub>"
                                    )
                                )
                            
                            # Add reference line if requested
                            if st.session_state.get('ref_line', False):
                                fig.add_hline(
                                    y=st.session_state.get('ref_line_val', 0),
                                    line_dash="dash",
                                    line_color="gray",
                                    annotation_text=st.session_state.get('ref_line_label', 'Reference'),
                                    annotation_position="right"
                                )
                            
                            # Add recession shading if requested
                            if st.session_state.get('recession_bars', False):
                                # Example recession periods (customize as needed)
                                recessions = [
                                    ('2020-02-01', '2020-04-01'),  # COVID-19
                                    ('2007-12-01', '2009-06-01'),  # Great Recession
                                ]
                                for start, end in recessions:
                                    fig.add_vrect(
                                        x0=start, x1=end,
                                        fillcolor="gray", opacity=0.2,
                                        layer="below", line_width=0,
                                        annotation_text="Recession",
                                        annotation_position="top left"
                                    )
                            
                            # Set chart dimensions
                            fig.update_layout(
                                width=chart_config.get('chart_width', 1000),
                                height=chart_config.get('chart_height', 600)
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
                            
                            # Export options
                            st.markdown("**💾 Export Chart**")
                            export_cols = st.columns(4)
                            with export_cols[0]:
                                if st.button("📥 PNG", key="export_png"):
                                    img_bytes = fig.to_image(format="png", width=chart_config.get('chart_width', 1000), height=chart_config.get('chart_height', 600))
                                    st.download_button("Download PNG", img_bytes, "chart.png", "image/png")
                            with export_cols[1]:
                                if st.button("📥 SVG", key="export_svg"):
                                    img_bytes = fig.to_image(format="svg", width=chart_config.get('chart_width', 1000), height=chart_config.get('chart_height', 600))
                                    st.download_button("Download SVG", img_bytes, "chart.svg", "image/svg+xml")
                            with export_cols[2]:
                                if st.button("📥 PDF", key="export_pdf"):
                                    img_bytes = fig.to_image(format="pdf", width=chart_config.get('chart_width', 1000), height=chart_config.get('chart_height', 600))
                                    st.download_button("Download PDF", img_bytes, "chart.pdf", "application/pdf")
                            with export_cols[3]:
                                if st.button("📥 Data", key="export_data"):
                                    csv = filtered_df.to_csv(index=False)
                                    st.download_button("Download CSV", csv, "chart_data.csv", "text/csv")
                            
                            st.plotly_chart(fig, use_container_width=False)
                            
                            # Save Chart inline
                            with st.expander("💾 Save Chart"):
                                save_col1, save_col2 = st.columns(2)
                                with save_col1:
                                    category = st.selectbox(
                                        "Category",
                                        ["Inflation", "Growth", "Employment", "Consumer", "Market", "Custom"],
                                        key="save_category"
                                    )
                                with save_col2:
                                    if st.button("💾 Save", key="save_chart_btn", use_container_width=True):
                                        try:
                                            report_config = chart_config.copy()
                                            report_config.update({
                                                'report_name': chart_config["title"],
                                                'report_category': category,
                                                'created_date': datetime.now().isoformat()
                                            })
                                            
                                            report_id = db.save_chart(
                                                chart_name=chart_config["title"][:50],
                                                chart_type="Line",
                                                figure_json=fig.to_json(),
                                                chart_config=report_config,
                                                user_prompt=f"Report: {chart_config['title']}",
                                                category=category
                                            )
                                            db.save_chart_data(report_id, 'report_data', filtered_df)
                                            st.success(f"✅ Saved! (ID: {report_id})")
                                            st.rerun()
                                        except Exception as e:
                                            st.error(f"Save failed: {str(e)[:50]}")
                            
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                else:
                    if series_col and value_col:
                        st.warning("Chart will be generated after pivot")
                    else:
                        st.warning("Please select at least one Y-axis column or configure series identifier")
    
    with col2:
        if st.session_state.data is not None:
            st.subheader("📊 Data Summary")
            
            if st.button("📊 View Data Summary", key="data_summary"):
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
            st.subheader("🤖 AI Analysis")
            
            # Show chart first
            chart = st.session_state.current_upload_chart
            st.plotly_chart(chart['figure'], use_container_width=True, key="upload_chart_display")
            
            # Quick Analysis
            if st.button("📊 Quick Analysis", key="quick_analysis_upload"):
                with st.spinner("Analyzing chart..."):
                    try:
                        chart_name = chart.get('config', {}).get('report_name', chart['prompt'])
                        category = chart.get('config', {}).get('report_category', 'Economic Analysis')
                        
                        analysis_prompt = f"""
Analyze this chart: {chart_name}

Provide 3 bullet points:
• Main trend
• Key insight
• Implication

Limit: 100 words.
"""
                        
                        analysis = llm_handler._call_llm(
                            "You are an economic analyst providing chart insights.",
                            analysis_prompt,
                            llm_provider
                        )
                        
                        st.markdown("### 🔍 Analysis")
                        st.markdown(analysis)
                        
                    except Exception as e:
                        st.markdown("### 🔍 Analysis")
                        st.markdown(f"**Chart**: {chart['prompt']}\n\n**Insight**: This economic visualization shows data patterns that indicate trends worth monitoring for policy and investment decisions.")
                        st.error(f"AI analysis unavailable: {str(e)[:50]}...")
            
            # Persona Summaries
            st.markdown("### 📝 Summaries")
            personas = ["Executive", "Economist", "General Public"]
            
            for persona in personas:
                if st.button(f"📝 {persona}", key=f"upload_summary_{persona.lower()}"):
                    with st.spinner(f"Generating {persona} summary..."):
                        try:
                            chart_name = chart.get('config', {}).get('report_name', chart['prompt'])
                            category = chart.get('config', {}).get('report_category', 'Economic Analysis')
                            
                            persona_prompt = f"""
As {persona}, analyze: {chart_name}

3 key points for {persona.lower()}:
• Main observation
• Decision impact
• Action item

Limit: 80 words.
"""
                            
                            summary = llm_handler._call_llm(
                                f"You are an expert {persona.lower()} providing economic analysis.",
                                persona_prompt,
                                llm_provider
                            )
                            
                            st.markdown(f"### 📄 {persona} Summary")
                            st.markdown(summary)
                            
                        except Exception as e:
                            st.markdown(f"### 📄 {persona} Summary")
                            st.markdown(f"**{persona} Analysis**: This economic chart shows data patterns relevant for {persona.lower()} decision-making and strategic planning.")
                            st.error(f"AI unavailable: {str(e)[:50]}...")
            
            # Q&A
            st.markdown("### ❓ Ask Questions")
            question = st.text_input("Your question:", key="qa_upload")
            
            if st.button("🔍 Get Answer", key="answer_upload"):
                if question:
                    with st.spinner("Getting answer..."):
                        try:
                            chart_name = chart.get('config', {}).get('report_name', chart['prompt'])
                            category = chart.get('config', {}).get('report_category', 'Economic Analysis')
                            
                            qa_prompt = f"""
Q: {question}
Chart: {chart_name}

Answer in 60 words.
"""
                            
                            answer = llm_handler._call_llm(
                                "You are an economic analyst answering questions about charts.",
                                qa_prompt,
                                llm_provider
                            )
                            
                            st.markdown("### 💬 Answer")
                            st.markdown(answer)
                            
                        except Exception as e:
                            st.markdown("### 💬 Answer")
                            chart_name = chart.get('config', {}).get('report_name', chart['prompt'])
                            st.markdown(f"**Question**: {question}\n\n**Response**: Based on the chart analysis for '{chart_name}', this economic data contains information that would require detailed analysis to provide a comprehensive answer.")
                            st.error(f"AI unavailable: {str(e)[:50]}...")

elif page == "saved":
    st.subheader("💾 Saved Charts")
    
    # Load saved charts
    try:
        all_charts = db.list_charts(limit=100)
        if all_charts:
            chart_names = [f"{c['chart_name']} (ID: {c['id']})" for c in all_charts]
            selected = st.selectbox("Select Chart", chart_names, key="saved_chart_select")
            
            if selected:
                chart_id = int(selected.split("ID: ")[1].rstrip(")"))
                saved_chart = db.get_chart(chart_id)
                
                if saved_chart:
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        # Display chart
                        fig = go.Figure(json.loads(saved_chart['figure_json']))
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.subheader("🤖 AI Analysis")
                        
                        chart_name = saved_chart['chart_name']
                        
                        # Quick Analysis
                        if st.button("📊 Quick Analysis", key="saved_quick_analysis"):
                            with st.spinner("Analyzing..."):
                                try:
                                    analysis_prompt = f"""
Analyze this chart: {chart_name}

Provide 3 bullet points:
• Main trend
• Key insight
• Implication

Limit: 100 words.
"""
                                    analysis = llm_handler._call_llm(
                                        "You are an economic analyst providing chart insights.",
                                        analysis_prompt,
                                        llm_provider
                                    )
                                    st.markdown("### 🔍 Analysis")
                                    st.markdown(analysis)
                                except Exception as e:
                                    st.error(f"AI unavailable: {str(e)[:50]}")
                        
                        # Persona Summaries
                        st.markdown("### 📝 Summaries")
                        personas = ["Executive", "Economist", "General Public"]
                        
                        for persona in personas:
                            if st.button(f"📝 {persona}", key=f"saved_summary_{persona.lower()}"):
                                with st.spinner(f"Generating {persona} summary..."):
                                    try:
                                        persona_prompt = f"""
As {persona}, analyze: {chart_name}

3 key points for {persona.lower()}:
• Main observation
• Decision impact
• Action item

Limit: 80 words.
"""
                                        summary = llm_handler._call_llm(
                                            f"You are an expert {persona.lower()} providing economic analysis.",
                                            persona_prompt,
                                            llm_provider
                                        )
                                        st.markdown(f"### 📄 {persona} Summary")
                                        st.markdown(summary)
                                    except Exception as e:
                                        st.error(f"AI unavailable: {str(e)[:50]}")
                        
                        # Q&A
                        st.markdown("### ❓ Ask Questions")
                        question = st.text_input("Your question:", key="saved_qa")
                        
                        if st.button("🔍 Get Answer", key="saved_answer"):
                            if question:
                                with st.spinner("Getting answer..."):
                                    try:
                                        qa_prompt = f"""
Q: {question}
Chart: {chart_name}

Answer in 60 words.
"""
                                        answer = llm_handler._call_llm(
                                            "You are an economic analyst answering questions about charts.",
                                            qa_prompt,
                                            llm_provider
                                        )
                                        st.markdown("### 💬 Answer")
                                        st.markdown(answer)
                                    except Exception as e:
                                        st.error(f"AI unavailable: {str(e)[:50]}")
        else:
            st.info("No saved charts yet. Create and save charts from the Upload Data page.")
    except Exception as e:
        st.error(f"Error loading charts: {str(e)}")

elif page == "image_upload":
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🖼️ Upload Chart Image & Data")
        
        # Image upload
        uploaded_image = st.file_uploader(
            "Upload chart image",
            type=['png', 'jpg', 'jpeg'],
            key="image_upload"
        )
        
        if uploaded_image:
            st.image(uploaded_image, caption="Uploaded Chart", use_column_width=True)
            # Convert to base64
            import base64
            img_base64 = base64.b64encode(uploaded_image.read()).decode()
            st.session_state.uploaded_image = img_base64
            st.success("✅ Image uploaded successfully!")
        
        # Data upload
        uploaded_data = st.file_uploader(
            "Upload data file (CSV/Excel)",
            type=['csv', 'xlsx', 'xls'],
            key="data_upload"
        )
        
        if uploaded_data:
            try:
                if uploaded_data.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_data)
                else:
                    df = pd.read_excel(uploaded_data)
                st.session_state.uploaded_data = df
                
                st.success(f"✅ Data loaded: {len(df)} rows, {len(df.columns)} columns")
                st.dataframe(df.head(10), use_container_width=True)
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
        
        # Metadata upload for context
        st.markdown("---")
        st.markdown("### 📝 Upload Chart Metadata (Optional)")
        chart_metadata_file = st.file_uploader(
            "Upload chart metadata (JSON/YAML)",
            type=['json', 'yaml', 'yml'],
            key="chart_metadata_upload",
            help="Provide context about the chart for better AI analysis"
        )
        
        if chart_metadata_file:
            try:
                file_content = chart_metadata_file.read().decode('utf-8')
                file_type = chart_metadata_file.name.split('.')[-1]
                chart_meta = metadata_handler.parse_metadata(file_content, file_type)
                st.session_state.uploaded_chart_metadata = chart_meta
                st.success("✅ Chart metadata loaded!")
                with st.expander("📋 View Chart Metadata"):
                    st.json(chart_meta)
            except Exception as e:
                st.error(f"Metadata error: {str(e)}")
                st.session_state.uploaded_chart_metadata = None
    
    with col2:
        # Show AI analysis options when both image and data are uploaded
        if 'uploaded_image' in st.session_state and 'uploaded_data' in st.session_state:
            st.subheader("🤖 AI Analysis")
            
            # Quick Analysis
            if st.button("📊 Analyze Chart & Data", key="analyze_uploaded"):
                with st.spinner("Analyzing chart and data..."):
                    try:
                        chart_metadata = st.session_state.get('uploaded_chart_metadata', None)
                        
                        # Build context from metadata
                        metadata_context = ""
                        if chart_metadata:
                            if 'chart_identity' in chart_metadata:
                                identity = chart_metadata['chart_identity']
                                metadata_context += f"\n\nChart Identity:\n- Title: {identity.get('title', 'N/A')}\n- Type: {identity.get('chart_type', 'N/A')}\n- Description: {identity.get('description', 'N/A')}"
                            
                            if 'data_configuration' in chart_metadata:
                                data_config = chart_metadata['data_configuration']
                                if 'data_source' in data_config:
                                    source = data_config['data_source']
                                    metadata_context += f"\n\nData Source:\n- Source: {source.get('primary_source', 'N/A')}\n- Frequency: {source.get('frequency', 'N/A')}\n- Transformation: {source.get('transformation', 'N/A')}"
                                
                                if 'components' in data_config:
                                    components = [c.get('name', '') for c in data_config['components']]
                                    metadata_context += f"\n\nComponents: {', '.join(components)}"
                        
                        analysis_prompt = f"""
You are an economic analyst. Analyze the provided chart image and dataset.

CHART CONTEXT:
The image shows an economic visualization with data trends and patterns.{metadata_context}

DATA SUMMARY:
- Dataset contains {len(st.session_state.uploaded_data)} rows and {len(st.session_state.uploaded_data.columns)} columns
- Key columns: {', '.join(st.session_state.uploaded_data.columns[:5])}
- Data ranges from {st.session_state.uploaded_data.iloc[0, 0] if len(st.session_state.uploaded_data) > 0 else 'N/A'} to {st.session_state.uploaded_data.iloc[-1, 0] if len(st.session_state.uploaded_data) > 0 else 'N/A'}

TASK:
Provide a concise economic analysis focusing on:
1. What the chart visualizes
2. Key trends and patterns
3. Economic implications
4. Notable insights

IMPORTANT: Limit response to 150-250 words. Do NOT repeat the raw data in your response.
"""
                        
                        analysis = llm_handler._call_llm(
                            "You are an economic analyst. Analyze the provided chart image and dataset.",
                            analysis_prompt,
                            llm_provider
                        )
                        
                        st.markdown("### 🔍 Analysis")
                        st.markdown(analysis)
                        
                    except Exception as e:
                        st.markdown("### 🔍 Analysis")
                        st.markdown(f"**Data**: {len(st.session_state.uploaded_data)} rows, {len(st.session_state.uploaded_data.columns)} columns\n\n**Insight**: Chart and data show economic patterns requiring analysis.")
                        st.error(f"AI unavailable: {str(e)[:50]}...")
            
            # Persona Summaries
            st.markdown("### 📝 Summaries")
            personas = ["Executive", "Economist", "General Public"]
            
            for persona in personas:
                if st.button(f"📝 {persona}", key=f"img_summary_{persona.lower()}"):
                    with st.spinner(f"Generating {persona} summary..."):
                        try:
                            chart_metadata = st.session_state.get('uploaded_chart_metadata', None)
                            
                            # Build metadata context
                            metadata_context = ""
                            if chart_metadata:
                                if 'chart_identity' in chart_metadata:
                                    identity = chart_metadata['chart_identity']
                                    metadata_context += f"\n\nChart: {identity.get('title', 'Economic Chart')}\nDescription: {identity.get('description', 'N/A')}"
                                
                                if 'data_configuration' in chart_metadata and 'data_source' in chart_metadata['data_configuration']:
                                    source = chart_metadata['data_configuration']['data_source']
                                    metadata_context += f"\nSource: {source.get('primary_source', 'N/A')}"
                            
                            persona_prompt = f"""
As a {persona}, analyze this economic chart and dataset.{metadata_context}

Chart: Economic visualization showing data trends and relationships.

Data Context:
- Dataset: {len(st.session_state.uploaded_data)} observations across {len(st.session_state.uploaded_data.columns)} variables
- Key metrics: {', '.join(st.session_state.uploaded_data.columns[:3])}
- Time period: {st.session_state.uploaded_data.iloc[0, 0] if len(st.session_state.uploaded_data) > 0 else 'N/A'} to {st.session_state.uploaded_data.iloc[-1, 0] if len(st.session_state.uploaded_data) > 0 else 'N/A'}

As a {persona.lower()}, provide analysis covering:
1. Chart interpretation from your perspective
2. Key economic insights
3. Implications for {persona.lower()} decision-making
4. Actionable recommendations
5. Risk considerations

Write 200-300 words maximum. Focus on insights, NOT raw data. Tailor language for {persona.lower()} audience.
"""
                            
                            summary = llm_handler._call_llm(
                                f"You are an expert {persona.lower()} providing economic analysis.",
                                persona_prompt,
                                llm_provider
                            )
                            
                            st.markdown(f"### 📄 {persona} Summary")
                            st.markdown(summary)
                            
                        except Exception as e:
                            st.markdown(f"### 📄 {persona} Summary")
                            st.markdown(f"**{persona} Analysis**: Chart and data provide insights for {persona.lower()} decision-making.")
                            st.error(f"AI unavailable: {str(e)[:50]}...")
            
            # Q&A
            st.markdown("### ❓ Ask Questions")
            question = st.text_input("Your question:", key="qa_img")
            
            if st.button("🔍 Get Answer", key="answer_img"):
                if question:
                    with st.spinner("Getting answer..."):
                        try:
                            chart_metadata = st.session_state.get('uploaded_chart_metadata', None)
                            
                            # Build metadata context
                            metadata_context = ""
                            if chart_metadata:
                                if 'chart_identity' in chart_metadata:
                                    identity = chart_metadata['chart_identity']
                                    metadata_context += f"\nChart: {identity.get('title', 'N/A')}\nType: {identity.get('chart_type', 'N/A')}"
                                
                                if 'data_schema' in chart_metadata:
                                    schema_info = [f"{k}: {v.get('description', '')}" for k, v in list(chart_metadata['data_schema'].items())[:3]]
                                    metadata_context += f"\nData Schema: {'; '.join(schema_info)}"
                            
                            qa_prompt = f"""
Question: {question}{metadata_context}

Chart Context:
Economic visualization showing data patterns and trends.

Data Summary:
- Dataset contains {len(st.session_state.uploaded_data)} records with {len(st.session_state.uploaded_data.columns)} variables
- Variables include: {', '.join(st.session_state.uploaded_data.columns)}
- Data spans: {st.session_state.uploaded_data.iloc[0, 0] if len(st.session_state.uploaded_data) > 0 else 'N/A'} to {st.session_state.uploaded_data.iloc[-1, 0] if len(st.session_state.uploaded_data) > 0 else 'N/A'}

Answer the question based on the chart visualization and data context.

Limit response to 100-200 words. Provide insights and analysis, NOT raw data values.
"""
                            
                            answer = llm_handler._call_llm(
                                "You are an economic analyst answering questions about charts.",
                                qa_prompt,
                                llm_provider
                            )
                            
                            st.markdown("### 💬 Answer")
                            st.markdown(answer)
                            
                        except Exception as e:
                            st.markdown("### 💬 Answer")
                            st.markdown(f"**Question**: {question}\n\n**Response**: Based on chart and data analysis.")
                            st.error(f"AI unavailable: {str(e)[:50]}...")
        else:
            st.info("ℹ️ Upload both image and data to enable AI analysis")

else:  # main page - show chart generation interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if 'current_chart' in st.session_state and st.session_state.current_chart:
            chart = st.session_state.current_chart
            st.subheader(f"📊 {chart['name']}")
            st.plotly_chart(chart['figure'], use_container_width=True)
            
            # AI Customization
            with st.expander("✨ AI Customization", expanded=False):
                custom_prompt = st.text_area(
                    "Describe changes",
                    placeholder="e.g., Change title to 'Recent Inflation Trends', use blue colors, render averages for GDP and inflation",
                    key="custom_existing"
                )
                
                if st.button("🔄 Apply Changes", key="apply_existing"):
                    if custom_prompt:
                        with st.spinner("Applying AI customization..."):
                            try:
                                # Get chart data for context
                                chart_data = None
                                if 'id' in chart:
                                    chart_data = db.get_chart_data(chart['id'], 'report_data')
                                
                                updated_config = llm_handler.interpret_edit_request(
                                    custom_prompt,
                                    {'chart_type': 'existing', 'title': chart['figure'].layout.title.text if chart['figure'].layout.title else chart['name']},
                                    llm_provider,
                                    chart_data
                                )
                                
                                if 'title' in updated_config:
                                    chart['figure'].update_layout(title=f"<b>{updated_config['title']}</b>")
                                
                                # Handle adding averages
                                if chart_data is not None and ('avg' in custom_prompt.lower() or 'average' in custom_prompt.lower()):
                                    numeric_cols = chart_data.select_dtypes(include='number').columns.tolist()
                                    for col in numeric_cols:
                                        if col in ['gdp_growth', 'inflation'] or any(term in col.lower() for term in ['gdp', 'inflation']):
                                            avg_val = chart_data[col].mean()
                                            chart['figure'].add_hline(
                                                y=avg_val,
                                                line_dash="dash",
                                                line_color="red" if 'inflation' in col.lower() else "blue",
                                                annotation_text=f"{col} avg: {avg_val:.2f}",
                                                annotation_position="top right"
                                            )
                                
                                # Save customized chart
                                chart_id = db.save_chart(
                                    chart_name=f"{chart['name']}_customized",
                                    chart_type="existing_custom",
                                    figure_json=chart['figure'].to_json(),
                                    chart_config=updated_config,
                                    user_prompt=custom_prompt
                                )
                                
                                st.success(f"✅ Customization applied and saved! (ID: {chart_id})")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error: {str(e)[:100]}...")
        else:
            st.info("👈 Select a chart from the sidebar filters and click Generate Chart")
    
    with col2:
        if 'current_chart' in st.session_state and st.session_state.current_chart:
            st.subheader("🤖 AI Wizard")
            
            # Analysis buttons
            if st.button("📊 Quick Analysis", key="quick_analysis"):
                with st.spinner("Analyzing chart..."):
                    try:
                        chart_name = st.session_state.current_chart['name']
                        
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
                            llm_provider
                        )
                        
                        st.markdown("### 🔍 Analysis")
                        st.markdown(analysis)
                    except Exception as e:
                        st.markdown("### 🔍 Analysis")
                        st.markdown(f"**Chart**: {st.session_state.current_chart['name']}\n\n**Insight**: This economic visualization shows data patterns that indicate trends worth monitoring for policy and investment decisions.")
                        st.error(f"Analysis failed: {str(e)[:100]}...")
            
            # Chart Customization
            st.markdown("### 🎨 Dynamic Chart Customization")
            
            # Initialize preview state
            if 'preview_figure' not in st.session_state:
                st.session_state.preview_figure = None
            if 'preview_config' not in st.session_state:
                st.session_state.preview_config = None
            
            custom_prompt = st.text_area(
                "Describe changes (updates in real-time)",
                placeholder="e.g., Change title to 'Recent Inflation Trends', use blue colors, render averages for GDP and inflation",
                key="custom_main"
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔍 Preview Changes", key="preview_main"):
                    if custom_prompt:
                        with st.spinner("Generating preview..."):
                            try:
                                # Get chart data for context
                                chart_data = None
                                if 'id' in st.session_state.current_chart:
                                    chart_data = db.get_chart_data(st.session_state.current_chart['id'], 'report_data')
                                
                                # Create context for customization
                                current_config = {
                                    'chart_type': 'existing',
                                    'title': st.session_state.current_chart['figure'].layout.title.text if st.session_state.current_chart['figure'].layout.title else st.session_state.current_chart['name'],
                                    'chart_name': st.session_state.current_chart['name']
                                }
                                
                                updated_config = llm_handler.interpret_edit_request(
                                    custom_prompt,
                                    current_config,
                                    llm_provider,
                                    chart_data
                                )
                                
                                # Create preview figure (copy of original)
                                import copy
                                preview_fig = copy.deepcopy(st.session_state.current_chart['figure'])
                                
                                # Apply customizations to preview
                                if 'title' in updated_config:
                                    preview_fig.update_layout(title=f"<b>{updated_config['title']}</b>")
                                
                                if 'colors' in updated_config:
                                    colors = updated_config['colors']
                                    if isinstance(colors, list):
                                        for i, trace in enumerate(preview_fig.data):
                                            if i < len(colors):
                                                trace.update(line=dict(color=colors[i]))
                                
                                if 'grid' in updated_config and updated_config['grid']:
                                    preview_fig.update_layout(
                                        xaxis=dict(showgrid=True),
                                        yaxis=dict(showgrid=True)
                                    )
                                
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
                                st.session_state.preview_figure = preview_fig
                                st.session_state.preview_config = updated_config
                                
                                st.success("✅ Preview generated! Review changes below.")
                                st.rerun()
                                
                            except Exception as e:
                                st.error(f"Preview failed: {str(e)[:100]}...")
            
            with col2:
                if st.button("💾 Apply & Save Changes", key="apply_main", disabled=st.session_state.preview_figure is None):
                    if st.session_state.preview_figure is not None:
                        with st.spinner("Saving changes..."):
                            try:
                                # Apply preview to main chart
                                st.session_state.current_chart['figure'] = st.session_state.preview_figure
                                
                                # Save to database
                                chart_id = db.save_chart(
                                    chart_name=f"{st.session_state.current_chart['name']}_custom",
                                    chart_type="filter_custom",
                                    figure_json=st.session_state.preview_figure.to_json(),
                                    chart_config=st.session_state.preview_config,
                                    user_prompt=custom_prompt
                                )
                                
                                # Clear preview
                                st.session_state.preview_figure = None
                                st.session_state.preview_config = None
                                
                                st.success(f"✅ Changes applied and saved! (ID: {chart_id})")
                                st.rerun()
                                
                            except Exception as e:
                                st.error(f"Save failed: {str(e)[:100]}...")
            
            # Show preview if available
            if st.session_state.preview_figure is not None:
                st.markdown("#### 🔍 Preview")
                st.info("👆 Review the changes above. Click 'Apply & Save Changes' to make them permanent.")
                st.plotly_chart(st.session_state.preview_figure, use_container_width=True, key="preview_chart")
                
                if st.button("🔄 Reset Preview", key="reset_preview"):
                    st.session_state.preview_figure = None
                    st.session_state.preview_config = None
                    st.rerun()
            
            # Persona summaries
            st.markdown("### 👥 Persona Summaries")
            personas = ["Executive", "Economist", "General Public"]
            
            for persona in personas:
                if st.button(f"📝 {persona}", key=f"summary_{persona.lower()}"):
                    with st.spinner(f"Generating {persona} summary..."):
                        try:
                            # Direct LLM call without generate_summary
                            chart_name = st.session_state.current_chart['name']
                            
                            persona_prompt = f"""
You are a {persona} analyzing this economic chart: {chart_name}

As a {persona.lower()}, provide analysis covering:
1. Key economic insights
2. Implications for {persona.lower()} decision-making
3. Actionable recommendations

Write 200-300 words maximum. Focus on insights.
"""
                            
                            summary = llm_handler._call_llm(
                                f"You are an expert {persona.lower()} providing economic analysis.",
                                persona_prompt,
                                llm_provider
                            )
                            
                            st.markdown(f"### 📄 {persona} Summary")
                            st.markdown(summary)
                            
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                            st.markdown(f"### 📄 {persona} Summary")
                            st.markdown(f"**{persona} Analysis**: This economic chart shows data patterns relevant for {persona.lower()} decision-making.")
            
            # Q&A
            st.markdown("### ❓ Ask Questions")
            question = st.text_input("Your question:", key="qa_existing")
            
            if st.button("🔍 Get Answer", key="answer_existing"):
                if question:
                    with st.spinner("Getting answer..."):
                        try:
                            chart_name = st.session_state.current_chart['name']
                            
                            qa_prompt = f"""
Question: {question}

Chart: {chart_name}

Answer the question about this economic chart. Provide insights based on typical economic analysis.

Limit response to 100-150 words.
"""
                            
                            answer = llm_handler._call_llm(
                                "You are an economic analyst answering questions about charts.",
                                qa_prompt,
                                llm_provider
                            )
                            
                            st.markdown("### 💬 Answer")
                            st.markdown(answer)
                        except Exception as e:
                            st.markdown("### 💬 Answer")
                            st.markdown(f"**Question**: {question}\n\n**Response**: Based on the chart data, this requires detailed analysis to provide a comprehensive answer.")
                            st.error(f"AI unavailable: {str(e)[:50]}...")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info("""
**Economic Chart Assistant**

Combined features:
- Pre-created PCE charts
- Custom chart creation
- AI-powered customization
- Date range filtering
- Chart summarization
- Image-based AI analysis

Using AWS Bedrock for AI
""")
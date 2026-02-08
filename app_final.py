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
                df[col] = pd.to_datetime(df[col]).dt.strftime('%Y-%m-%d')
                date_cols.append(col)
            except:
                pass
    
    # If no date columns found, try first column
    if not date_cols and len(df.columns) > 0:
        try:
            df[df.columns[0]] = pd.to_datetime(df[df.columns[0]]).dt.strftime('%Y-%m-%d')
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

# Filters
st.sidebar.markdown("### 🔍 Filters")

# Category filter
categories = {
    "PCE Inflation": [k for k in CHART_REGISTRY.keys() if k.startswith('01') or k.startswith('02')],
    "PCE Components": [k for k in CHART_REGISTRY.keys() if k.startswith('03')],
    "Timing Analysis": [k for k in CHART_REGISTRY.keys() if k.startswith('04')],
    "Decomposition": [k for k in CHART_REGISTRY.keys() if k.startswith('05')],
    "Supply/Demand": [k for k in CHART_REGISTRY.keys() if k.startswith('fig7')]
}

selected_category = st.sidebar.selectbox(
    "Category",
    list(categories.keys()),
    key="category_filter"
)

# Chart filter within category
available_charts = categories[selected_category]
selected_chart = st.sidebar.selectbox(
    "Chart",
    available_charts,
    key="chart_filter"
)

# Chart type filter
chart_types = ["Line", "Bar", "Pie", "Stacked Bar", "Area", "Scatter", "Heatmap"]
selected_chart_type = st.sidebar.selectbox(
    "Chart Type",
    chart_types,
    key="chart_type_filter"
)

# Date range
st.sidebar.markdown("### 📅 Date Range")
try:
    pce_df = pd.read_csv("charting_assistant/test_pce_data.csv")
    pce_df['date'] = pd.to_datetime(pce_df['date'])
    min_date = pce_df['date'].min().date()
    max_date = pce_df['date'].max().date()
except:
    min_date = pd.Timestamp('2020-01-01').date()
    max_date = pd.Timestamp.now().date()

start_date = st.sidebar.date_input("Start Date", value=min_date, key="start_filter")
end_date = st.sidebar.date_input("End Date", value=max_date, key="end_filter")

if st.sidebar.button("🚀 Generate Chart", type="primary", key="gen_selected"):
    with st.spinner(f"Generating {selected_chart}..."):
        try:
            fig = generate_chart(selected_chart, "charting_assistant/test_pce_data.csv", "charting_assistant/test_supply_demand_data.csv")
            st.session_state.current_chart = {
                'name': selected_chart,
                'figure': fig,
                'category': selected_category,
                'chart_type': selected_chart_type,
                'date_range': [start_date, end_date]
            }
            # Clear other states and navigate to main
            if 'current_upload_chart' in st.session_state:
                del st.session_state.current_upload_chart
            if 'uploaded_image' in st.session_state:
                del st.session_state.uploaded_image
            if 'uploaded_data' in st.session_state:
                del st.session_state.uploaded_data
            st.session_state.page = "main"
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Error: {str(e)}")

st.sidebar.markdown("---")
st.sidebar.markdown("### Settings")
llm_provider = st.sidebar.selectbox(
    "LLM Provider",
    ["AWS Bedrock", "OpenAI", "Anthropic"]
)

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
                        date_series = pd.to_datetime(df[x_column])
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
                                    date_series = pd.to_datetime(filtered_df[x_column])
                                    mask = (date_series.dt.date >= start_date) & (date_series.dt.date <= end_date)
                                    filtered_df = filtered_df[mask]
                                except:
                                    pass
                            
                            # Create chart config with auto ranges
                            chart_config = {
                                "chart_type": "line",
                                "x_column": x_column,
                                "y_columns": y_columns,
                                "title": user_prompt if user_prompt else f"{', '.join(y_columns)} over {x_column}",
                                "x_label": x_column,
                                "y_label": ", ".join(y_columns),  # Show metrics instead of series names
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
                                'prompt': user_prompt or f"Chart: {', '.join(y_columns)} vs {x_column}"
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
            
            # Customization
            custom_prompt = st.text_area(
                "Customize chart",
                placeholder="Change colors to blue, add title",
                key="custom_upload"
            )
            
            if st.button("🔄 Apply Changes", key="apply_upload"):
                if custom_prompt:
                    with st.spinner("Applying changes..."):
                        try:
                            chart = st.session_state.current_upload_chart
                            updated_config = llm_handler.interpret_edit_request(
                                custom_prompt,
                                chart['config'],
                                llm_provider
                            )
                            
                            fig = chart_generator.create_chart(st.session_state.data, updated_config)
                            
                            # Update saved chart
                            db.update_chart(
                                chart['id'],
                                chart_config=updated_config,
                                figure_json=fig.to_json()
                            )
                            
                            st.session_state.current_upload_chart['figure'] = fig
                            st.session_state.current_upload_chart['config'] = updated_config
                            
                            st.success("✅ Chart updated!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
            
            # Quick Analysis with Image
            if st.button("📊 Quick Analysis", key="quick_analysis_upload"):
                with st.spinner("Analyzing chart..."):
                    try:
                        # Simple analysis without image generation
                        chart_name = st.session_state.current_upload_chart['prompt']
                        
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
                        st.markdown(f"**Chart**: {st.session_state.current_upload_chart['prompt']}\n\n**Insight**: This economic visualization shows data patterns that indicate trends worth monitoring for policy and investment decisions.")
                        st.error(f"AI analysis unavailable: {str(e)[:50]}...")
            
            # Persona Summaries with Image
            st.markdown("### 📝 Summaries")
            personas = ["Executive", "Economist", "General Public"]
            
            for persona in personas:
                if st.button(f"📝 {persona}", key=f"upload_summary_{persona.lower()}"):
                    with st.spinner(f"Generating {persona} summary..."):
                        try:
                            # Simple persona analysis without chart context generation
                            chart_name = st.session_state.current_upload_chart['prompt']
                            
                            persona_prompt = f"""
As a {persona}, analyze this economic chart: {chart_name}

Provide analysis covering:
1. Key economic insights from your {persona.lower()} perspective
2. Implications for {persona.lower()} decision-making
3. Actionable recommendations

Write 200-250 words maximum. Focus on insights relevant to a {persona.lower()}.
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
            
            # Q&A with simplified approach
            st.markdown("### ❓ Ask Questions")
            question = st.text_input("Your question:", key="qa_upload")
            
            if st.button("🔍 Get Answer", key="answer_upload"):
                if question:
                    with st.spinner("Getting answer..."):
                        try:
                            chart_name = st.session_state.current_upload_chart['prompt']
                            
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
                            st.markdown(f"**Question**: {question}\n\n**Response**: Based on the chart analysis, this economic data contains information that would require detailed analysis to provide a comprehensive answer.")
                            st.error(f"AI unavailable: {str(e)[:50]}...")

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
    
    with col2:
        # Show AI analysis options when both image and data are uploaded
        if 'uploaded_image' in st.session_state and 'uploaded_data' in st.session_state:
            st.subheader("🤖 AI Analysis")
            
            # Quick Analysis
            if st.button("📊 Analyze Chart & Data", key="analyze_uploaded"):
                with st.spinner("Analyzing chart and data..."):
                    try:
                        analysis_prompt = f"""
You are an economic analyst. Analyze the provided chart image and dataset.

CHART CONTEXT:
The image shows an economic visualization with data trends and patterns.

DATA SUMMARY:
- Dataset contains {len(st.session_state.uploaded_data)} rows and {len(st.session_state.uploaded_data.columns)} columns
- Key columns: {', '.join(st.session_state.uploaded_data.columns[:5])}
- Data ranges from {st.session_state.uploaded_data.iloc[0, 0] if len(st.session_state.uploaded_data) > 0 else 'N/A'} to {st.session_state.uploaded_data.iloc[-1, 0] if len(st.session_state.uploaded_data) > 0 else 'N/A'}
- Sample values: {st.session_state.uploaded_data.iloc[0].to_dict() if len(st.session_state.uploaded_data) > 0 else 'N/A'}

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
                            persona_prompt = f"""
As a {persona}, analyze this economic chart and dataset.

Chart: Economic visualization showing data trends and relationships.

Data Context:
- Dataset: {len(st.session_state.uploaded_data)} observations across {len(st.session_state.uploaded_data.columns)} variables
- Key metrics: {', '.join(st.session_state.uploaded_data.columns[:3])}
- Time period: {st.session_state.uploaded_data.iloc[0, 0] if len(st.session_state.uploaded_data) > 0 else 'N/A'} to {st.session_state.uploaded_data.iloc[-1, 0] if len(st.session_state.uploaded_data) > 0 else 'N/A'}
- Recent values: {dict(list(st.session_state.uploaded_data.iloc[-1].items())[:3]) if len(st.session_state.uploaded_data) > 0 else 'N/A'}

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
                            qa_prompt = f"""
Question: {question}

Chart Context:
Economic visualization showing data patterns and trends.

Data Summary:
- Dataset contains {len(st.session_state.uploaded_data)} records with {len(st.session_state.uploaded_data.columns)} variables
- Variables include: {', '.join(st.session_state.uploaded_data.columns)}
- Data spans: {st.session_state.uploaded_data.iloc[0, 0] if len(st.session_state.uploaded_data) > 0 else 'N/A'} to {st.session_state.uploaded_data.iloc[-1, 0] if len(st.session_state.uploaded_data) > 0 else 'N/A'}
- Latest observation: {dict(list(st.session_state.uploaded_data.iloc[-1].items())[:3]) if len(st.session_state.uploaded_data) > 0 else 'N/A'}

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
                    placeholder="e.g., Change title to 'Recent Inflation Trends', use blue colors",
                    key="custom_existing"
                )
                
                if st.button("🔄 Apply Changes", key="apply_existing"):
                    if custom_prompt:
                        with st.spinner("Applying AI customization..."):
                            try:
                                updated_config = llm_handler.interpret_edit_request(
                                    custom_prompt,
                                    {'chart_type': 'existing', 'title': chart['figure'].layout.title.text if chart['figure'].layout.title else chart['name']},
                                    llm_provider
                                )
                                
                                if 'title' in updated_config:
                                    chart['figure'].update_layout(title=f"<b>{updated_config['title']}</b>")
                                
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
            st.markdown("### 🎨 Customize Chart")
            custom_prompt = st.text_area(
                "Describe changes",
                placeholder="Change title to 'Recent Inflation Trends', use blue colors, add grid lines",
                key="custom_main"
            )
            
            if st.button("🔄 Apply Changes", key="apply_main"):
                if custom_prompt:
                    with st.spinner("Applying customization..."):
                        try:
                            # Create context for customization
                            current_config = {
                                'chart_type': 'existing',
                                'title': st.session_state.current_chart['figure'].layout.title.text if st.session_state.current_chart['figure'].layout.title else st.session_state.current_chart['name'],
                                'chart_name': st.session_state.current_chart['name']
                            }
                            
                            updated_config = llm_handler.interpret_edit_request(
                                custom_prompt,
                                current_config,
                                llm_provider
                            )
                            
                            # Apply customizations to the figure
                            fig = st.session_state.current_chart['figure']
                            
                            if 'title' in updated_config:
                                fig.update_layout(title=f"<b>{updated_config['title']}</b>")
                            
                            if 'colors' in updated_config:
                                colors = updated_config['colors']
                                if isinstance(colors, list):
                                    for i, trace in enumerate(fig.data):
                                        if i < len(colors):
                                            trace.update(line=dict(color=colors[i]))
                            
                            if 'grid' in updated_config and updated_config['grid']:
                                fig.update_layout(
                                    xaxis=dict(showgrid=True),
                                    yaxis=dict(showgrid=True)
                                )
                            
                            # Save customized chart
                            chart_id = db.save_chart(
                                chart_name=f"{st.session_state.current_chart['name']}_custom",
                                chart_type="filter_custom",
                                figure_json=fig.to_json(),
                                chart_config=updated_config,
                                user_prompt=custom_prompt
                            )
                            
                            st.success(f"✅ Chart customized and saved! (ID: {chart_id})")
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"Customization failed: {str(e)[:100]}...")
            
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
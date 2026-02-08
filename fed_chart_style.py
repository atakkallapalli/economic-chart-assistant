"""
Federal Reserve Chart Style Configuration
Based on Federal Reserve Economic Data (FRED) and Fed publications styling
"""
import plotly.graph_objects as go
from typing import Dict, List, Optional
import plotly.io as pio
import numpy as np

# Federal Reserve Color Palette
FED_COLORS = {
    # Primary colors (from FRED)
    'primary_blue': '#4472C4',
    'primary_red': '#C5504B',
    'primary_green': '#70AD47',
    'primary_orange': '#FFC000',
    'primary_purple': '#7030A0',
    'primary_teal': '#00B0F0',
    
    # Secondary colors
    'secondary_blue': '#5B9BD5',
    'secondary_red': '#ED7D31',
    'secondary_green': '#A5A5A5',
    'secondary_yellow': '#FFC000',
    'secondary_purple': '#9E480E',
    'secondary_teal': '#255E91',
    
    # Neutral colors
    'dark_gray': '#404040',
    'medium_gray': '#808080',
    'light_gray': '#D9D9D9',
    'very_light_gray': '#F2F2F2',
    
    # Special purpose
    'target_line': '#808080',  # For Fed 2% target
    'recession_shade': '#E8E8E8',  # NBER recession shading
    'zero_line': '#000000',
    
    # Inflation specific
    'headline_inflation': '#4472C4',
    'core_inflation': '#C5504B',
    'food': '#FFC000',
    'energy': '#ED7D31',
    'goods': '#5B9BD5',
    'services': '#70AD47',
    'housing': '#7030A0',
    'supercore': '#9E480E',
}

# Color sequences for multiple series
FED_COLOR_SEQUENCES = {
    'default': [
        FED_COLORS['primary_blue'],
        FED_COLORS['primary_red'],
        FED_COLORS['primary_green'],
        FED_COLORS['primary_orange'],
        FED_COLORS['primary_purple'],
        FED_COLORS['primary_teal'],
    ],
    'inflation': [
        FED_COLORS['headline_inflation'],
        FED_COLORS['core_inflation'],
        FED_COLORS['food'],
        FED_COLORS['energy'],
    ],
    'components': [
        FED_COLORS['goods'],
        FED_COLORS['services'],
        FED_COLORS['housing'],
        FED_COLORS['supercore'],
    ],
    'shades_blue': ['#08519c', '#3182bd', '#6baed6', '#9ecae1', '#c6dbef'],
    'shades_red': ['#a50f15', '#de2d26', '#fb6a4a', '#fc9272', '#fcbba1'],
}

# Typography (Federal Reserve style)
FED_FONTS = {
    'title': {
        'family': 'Arial, sans-serif',
        'size': 16,
        'color': FED_COLORS['dark_gray']
    },
    'axis_title': {
        'family': 'Arial, sans-serif',
        'size': 12,
        'color': FED_COLORS['dark_gray'],
    },
    'axis_labels': {
        'family': 'Arial, sans-serif',
        'size': 10,
        'color': FED_COLORS['medium_gray'],
    },
    'legend': {
        'family': 'Arial, sans-serif',
        'size': 10,
        'color': FED_COLORS['dark_gray'],
    },
    'annotation': {
        'family': 'Arial, sans-serif',
        'size': 9,
        'color': FED_COLORS['medium_gray'],
    }
}

# Chart dimensions and margins (Federal Reserve standard)
FED_LAYOUT = {
    'width': 800,
    'height': 500,
    'margin': {
        't': 80,   # Top margin for title
        'r': 40,   # Right margin
        'b': 60,   # Bottom margin for x-axis
        'l': 80,   # Left margin for y-axis
        'pad': 4
    }
}

# Grid and axis styling
FED_GRID_STYLE = {
    'showgrid': True,
    'gridwidth': 1,
    'gridcolor': FED_COLORS['light_gray'],
    'zeroline': True,
    'zerolinewidth': 1.5,
    'zerolinecolor': FED_COLORS['zero_line'],
    'showline': True,
    'linewidth': 1,
    'linecolor': FED_COLORS['medium_gray'],
}

# Line styles
FED_LINE_STYLES = {
    'primary': {'width': 2.5, 'dash': 'solid'},
    'secondary': {'width': 2, 'dash': 'solid'},
    'tertiary': {'width': 1.5, 'dash': 'solid'},
    'target': {'width': 2, 'dash': 'dash'},
    'forecast': {'width': 2, 'dash': 'dot'},
    'reference': {'width': 1, 'dash': 'dashdot'},
}

# Marker styles
FED_MARKER_STYLES = {
    'default': {'size': 6, 'symbol': 'circle'},
    'emphasis': {'size': 8, 'symbol': 'circle'},
    'small': {'size': 4, 'symbol': 'circle'},
}


def create_fed_template() -> dict:
    """Create a Plotly template matching Federal Reserve style"""
    
    template = {
        'layout': {
            'font': {
                'family': FED_FONTS['axis_labels']['family'],
                'size': FED_FONTS['axis_labels']['size'],
                'color': FED_FONTS['axis_labels']['color']
            },
            'title': {
                'font': FED_FONTS['title'],
                'x': 0.05,
                'xanchor': 'left',
                'y': 0.95,
                'yanchor': 'top'
            },
            'xaxis': {
                **FED_GRID_STYLE,
                'title': {'font': FED_FONTS['axis_title']},
                'tickfont': FED_FONTS['axis_labels'],
            },
            'yaxis': {
                **FED_GRID_STYLE,
                'title': {'font': FED_FONTS['axis_title']},
                'tickfont': FED_FONTS['axis_labels'],
            },
            'legend': {
                'font': FED_FONTS['legend'],
                'bgcolor': 'rgba(255, 255, 255, 0.8)',
                'bordercolor': FED_COLORS['light_gray'],
                'borderwidth': 1,
                'x': 1.02,
                'xanchor': 'left',
                'y': 1,
                'yanchor': 'top'
            },
            'plot_bgcolor': 'white',
            'paper_bgcolor': 'white',
            'colorway': FED_COLOR_SEQUENCES['default'],
            'hovermode': 'x unified',
            'hoverlabel': {
                'bgcolor': 'white',
                'font': {'size': 10, 'family': 'Arial'},
                'bordercolor': FED_COLORS['medium_gray']
            },
            **FED_LAYOUT
        }
    }
    
    return template


def apply_fed_style(fig: go.Figure, 
                    title: Optional[str] = None,
                    subtitle: Optional[str] = None,
                    source: Optional[str] = None,
                    add_target_line: bool = False,
                    target_value: float = 2.0,
                    add_recession_shading: bool = False,
                    recession_periods: Optional[List[tuple]] = None) -> go.Figure:
    """
    Apply Federal Reserve styling to a Plotly figure
    
    Args:
        fig: Plotly figure object
        title: Chart title
        subtitle: Chart subtitle
        source: Data source attribution
        add_target_line: Add Fed 2% target line
        target_value: Target line value (default 2.0%)
        add_recession_shading: Add NBER recession shading
        recession_periods: List of (start_date, end_date) tuples for recessions
    """
    
    # Apply template
    template = create_fed_template()
    fig.update_layout(template['layout'])
    
    # Update title with subtitle if provided
    if title:
        title_text = f"<b>{title}</b>"
        if subtitle:
            title_text += f"<br><sub>{subtitle}</sub>"
        fig.update_layout(title={'text': title_text})
    
    # Add source attribution
    if source:
        fig.add_annotation(
            text=f"Source: {source}",
            xref="paper", yref="paper",
            x=0, y=-0.15,
            xanchor='left', yanchor='top',
            font=FED_FONTS['annotation'],
            showarrow=False
        )
    
    # Add Fed target line (typically 2% for inflation)
    if add_target_line:
        fig.add_hline(
            y=target_value,
            line_dash="dash",
            line_color=FED_COLORS['target_line'],
            line_width=FED_LINE_STYLES['target']['width'],
            annotation_text=f"{target_value}% Target",
            annotation_position="right",
            annotation_font=FED_FONTS['annotation']
        )
    
    # Add recession shading
    if add_recession_shading and recession_periods:
        for start, end in recession_periods:
            fig.add_vrect(
                x0=start, x1=end,
                fillcolor=FED_COLORS['recession_shade'],
                opacity=0.5,
                layer="below",
                line_width=0,
                annotation_text="Recession",
                annotation_position="top left",
                annotation_font=FED_FONTS['annotation']
            )
    
    return fig


def get_color_scheme(scheme_name: str = 'default') -> List[str]:
    """Get a color scheme by name"""
    return FED_COLOR_SEQUENCES.get(scheme_name, FED_COLOR_SEQUENCES['default'])


def get_color(color_name: str) -> str:
    """Get a specific color by name"""
    return FED_COLORS.get(color_name, FED_COLORS['primary_blue'])


# NBER Recession dates (major recessions since 2000)
NBER_RECESSIONS = [
    ('2001-03-01', '2001-11-01'),  # Dot-com bubble
    ('2007-12-01', '2009-06-01'),  # Great Recession
    ('2020-02-01', '2020-04-01'),  # COVID-19
]


# Register the Fed template globally
pio.templates['fed_style'] = create_fed_template()
pio.templates.default = 'fed_style'


# Example usage functions
def create_inflation_chart(x_data, y_data_headline, y_data_core, 
                          title: str = "PCE Inflation") -> go.Figure:
    """Create a standard Fed-style inflation chart"""
    
    fig = go.Figure()
    
    # Add headline inflation
    fig.add_trace(go.Scatter(
        x=x_data,
        y=y_data_headline,
        name='Headline PCE',
        line=dict(
            color=FED_COLORS['headline_inflation'],
            **FED_LINE_STYLES['primary']
        ),
        mode='lines'
    ))
    
    # Add core inflation
    fig.add_trace(go.Scatter(
        x=x_data,
        y=y_data_core,
        name='Core PCE',
        line=dict(
            color=FED_COLORS['core_inflation'],
            **FED_LINE_STYLES['primary']
        ),
        mode='lines'
    ))
    
    # Apply Fed styling
    fig = apply_fed_style(
        fig,
        title=title,
        subtitle="Year-over-year percent change",
        source="Bureau of Economic Analysis",
        add_target_line=True,
        target_value=2.0,
        add_recession_shading=True,
        recession_periods=NBER_RECESSIONS
    )
    
    fig.update_yaxes(title_text="Percent")
    fig.update_xaxes(title_text="Date")
    
    return fig


def create_decomposition_chart(x_data, components_data: Dict[str, list],
                               title: str = "PCE Components") -> go.Figure:
    """Create a stacked bar chart for component decomposition"""
    
    fig = go.Figure()
    
    colors = get_color_scheme('components')
    
    for idx, (name, data) in enumerate(components_data.items()):
        fig.add_trace(go.Bar(
            x=x_data,
            y=data,
            name=name,
            marker_color=colors[idx % len(colors)]
        ))
    
    fig.update_layout(barmode='stack')
    
    fig = apply_fed_style(
        fig,
        title=title,
        subtitle="Contribution to year-over-year change",
        source="Bureau of Economic Analysis"
    )
    
    fig.update_yaxes(title_text="Percentage points")
    fig.update_xaxes(title_text="Date")
    
    return fig


# Chart boundary configurations
CHART_BOUNDARIES = {
    'inflation': {
        'y_min': -2,
        'y_max': 10,
        'y_tick_interval': 1
    },
    'inflation_mom': {
        'y_min': -1,
        'y_max': 2,
        'y_tick_interval': 0.5
    },
    'growth_rate': {
        'y_min': -10,
        'y_max': 10,
        'y_tick_interval': 2
    },
    'index': {
        'y_min': 80,
        'y_max': 120,
        'y_tick_interval': 10
    },
    'percentage': {
        'y_min': 0,
        'y_max': 100,
        'y_tick_interval': 10
    }
}


def calculate_axis_range(data_values, padding_percent: float = 0.1):
    """
    Calculate axis range based on data min/max with padding
    
    Args:
        data_values: List or array of numeric values
        padding_percent: Percentage of range to add as padding (default 10%)
    
    Returns:
        tuple: (min_value, max_value) with padding applied
    """
    import numpy as np
    
    # Filter out None and NaN values
    clean_values = [v for v in data_values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    
    if not clean_values:
        return (0, 10)  # Default range if no valid data
    
    data_min = min(clean_values)
    data_max = max(clean_values)
    
    # Calculate range and padding
    data_range = data_max - data_min
    
    # Handle case where all values are the same
    if data_range == 0:
        padding = abs(data_min) * 0.1 if data_min != 0 else 1
        return (data_min - padding, data_max + padding)
    
    padding = data_range * padding_percent
    
    # Apply padding
    range_min = data_min - padding
    range_max = data_max + padding
    
    # Round to nice numbers
    def round_to_nice(value, round_up=True):
        """Round to nice numbers for axis labels"""
        if value == 0:
            return 0
        
        magnitude = 10 ** np.floor(np.log10(abs(value)))
        normalized = value / magnitude
        
        if round_up:
            nice_values = [1, 2, 5, 10]
            nice = min([v for v in nice_values if v >= normalized], default=10)
        else:
            nice_values = [1, 2, 5, 10]
            nice = max([v for v in nice_values if v <= normalized], default=1)
        
        return nice * magnitude if value > 0 else -nice * magnitude
    
    range_min = round_to_nice(range_min, round_up=False)
    range_max = round_to_nice(range_max, round_up=True)
    
    return (range_min, range_max)


def set_chart_boundaries(fig: go.Figure, boundary_type: str = 'inflation', auto_range: bool = True) -> go.Figure:
    """
    Set chart boundaries based on chart type or auto-calculate from data
    
    Args:
        fig: Plotly figure object
        boundary_type: Type of chart ('inflation', 'growth', etc.)
        auto_range: If True, calculate range from data; if False, use predefined boundaries
    """
    
    if auto_range:
        # Extract all y-values from traces
        all_y_values = []
        for trace in fig.data:
            if hasattr(trace, 'y') and trace.y is not None:
                all_y_values.extend([y for y in trace.y if y is not None])
        
        if all_y_values:
            y_min, y_max = calculate_axis_range(all_y_values)
            fig.update_yaxes(range=[y_min, y_max])
            
            # Calculate nice tick interval
            y_range = y_max - y_min
            if y_range > 0:
                # Aim for about 5-10 ticks
                tick_interval = y_range / 8
                # Round to nice number
                magnitude = 10 ** np.floor(np.log10(tick_interval))
                nice_ticks = [0.1, 0.2, 0.5, 1, 2, 5, 10]
                normalized = tick_interval / magnitude
                nice_tick = min([t for t in nice_ticks if t >= normalized], default=1)
                tick_interval = nice_tick * magnitude
                
                fig.update_yaxes(dtick=tick_interval)
    else:
        # Use predefined boundaries
        if boundary_type in CHART_BOUNDARIES:
            bounds = CHART_BOUNDARIES[boundary_type]
            fig.update_yaxes(
                range=[bounds['y_min'], bounds['y_max']],
                dtick=bounds['y_tick_interval']
            )
    
    return fig

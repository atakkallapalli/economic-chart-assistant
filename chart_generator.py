import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, Any

class ChartGenerator:
    """Generate professional-looking charts based on configuration"""
    
    # Professional color palettes
    PROFESSIONAL_COLORS = {
        'business': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'],
        'economic': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#592E83', '#1B998B'],
        'federal': ['#003f5c', '#58508d', '#bc5090', '#ff6361', '#ffa600', '#488f31'],
        'minimal': ['#4472C4', '#E70000', '#70AD47', '#FFC000', '#7030A0', '#C55A11']
    }
    
    def create_chart(self, df: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """Create a professional chart based on configuration"""
        
        chart_type = config.get('chart_type', 'line').lower()
        
        if chart_type == 'line':
            return self._create_line_chart(df, config)
        elif chart_type == 'bar':
            return self._create_bar_chart(df, config)
        elif chart_type == 'scatter':
            return self._create_scatter_chart(df, config)
        elif chart_type == 'area':
            return self._create_area_chart(df, config)
        elif chart_type == 'stacked_bar':
            return self._create_stacked_bar_chart(df, config)
        elif chart_type == 'stacked_area':
            return self._create_stacked_area_chart(df, config)
        elif chart_type == 'pie':
            return self._create_pie_chart(df, config)
        elif chart_type == 'heatmap':
            return self._create_heatmap(df, config)
        else:
            raise ValueError(f"Unsupported chart type: {chart_type}")
    
    def _get_colors(self, config: Dict, num_colors: int) -> list:
        """Get professional color palette"""
        if 'colors' in config and config['colors']:
            return config['colors']
        
        palette = config.get('color_palette', 'business')
        colors = self.PROFESSIONAL_COLORS.get(palette, self.PROFESSIONAL_COLORS['business'])
        
        # Extend colors if needed
        while len(colors) < num_colors:
            colors.extend(colors)
        
        return colors[:num_colors]
    
    def _apply_professional_styling(self, fig: go.Figure, config: Dict) -> go.Figure:
        """Apply professional styling to the chart"""
        
        # Professional layout
        fig.update_layout(
            title={
                'text': config.get('title', 'Chart'),
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'family': 'Arial, sans-serif', 'color': '#2E2E2E'}
            },
            font={'family': 'Arial, sans-serif', 'size': 12, 'color': '#2E2E2E'},
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=config.get('show_legend', True),
            legend={
                'orientation': 'h',
                'yanchor': 'bottom',
                'y': -0.2,
                'xanchor': 'center',
                'x': 0.5,
                'font': {'size': 11}
            },
            margin={'l': 60, 'r': 40, 't': 80, 'b': 80},
            hovermode='x unified'
        )
        
        # Professional axis styling
        fig.update_xaxes(
            title_font={'size': 14, 'color': '#2E2E2E'},
            tickfont={'size': 11, 'color': '#2E2E2E'},
            gridcolor='#E5E5E5',
            gridwidth=1,
            showgrid=True,
            zeroline=False,
            linecolor='#CCCCCC',
            linewidth=1
        )
        
        fig.update_yaxes(
            title_font={'size': 14, 'color': '#2E2E2E'},
            tickfont={'size': 11, 'color': '#2E2E2E'},
            gridcolor='#E5E5E5',
            gridwidth=1,
            showgrid=True,
            zeroline=True,
            zerolinecolor='#CCCCCC',
            zerolinewidth=1,
            linecolor='#CCCCCC',
            linewidth=1
        )
        
        # Add annotations if specified
        if 'annotations' in config and config['annotations']:
            for annotation in config['annotations']:
                fig.add_annotation(
                    x=annotation.get('x'),
                    y=annotation.get('y'),
                    text=annotation.get('text'),
                    showarrow=annotation.get('showarrow', True),
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor='#636363',
                    font={'size': 10, 'color': '#2E2E2E'}
                )
        
        return fig
    
    def _create_line_chart(self, df: pd.DataFrame, config: Dict) -> go.Figure:
        """Create a professional line chart"""
        x_col = config.get('x_column')
        y_cols = config.get('y_columns', [])
        
        fig = go.Figure()
        colors = self._get_colors(config, len(y_cols))
        
        for idx, y_col in enumerate(y_cols):
            if y_col in df.columns:
                # Professional line styling
                fig.add_trace(go.Scatter(
                    x=df[x_col],
                    y=df[y_col],
                    mode='lines+markers',
                    name=y_col.replace('_', ' ').title(),
                    line=dict(
                        color=colors[idx],
                        width=2.5
                    ),
                    marker=dict(
                        size=4,
                        color=colors[idx],
                        line=dict(width=1, color='white')
                    ),
                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                  f'{x_col}: %{{x}}<br>' +
                                  f'{y_col}: %{{y:.2f}}<extra></extra>'
                ))
        
        fig.update_layout(
            xaxis_title=config.get('x_label', x_col.replace('_', ' ').title()),
            yaxis_title=config.get('y_label', 'Value')
        )
        
        return self._apply_professional_styling(fig, config)
    
    def _create_bar_chart(self, df: pd.DataFrame, config: Dict) -> go.Figure:
        """Create a professional bar chart"""
        x_col = config.get('x_column')
        y_cols = config.get('y_columns', [])
        
        fig = go.Figure()
        colors = self._get_colors(config, len(y_cols))
        
        for idx, y_col in enumerate(y_cols):
            if y_col in df.columns:
                fig.add_trace(go.Bar(
                    x=df[x_col],
                    y=df[y_col],
                    name=y_col.replace('_', ' ').title(),
                    marker=dict(
                        color=colors[idx],
                        line=dict(color='white', width=1)
                    ),
                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                  f'{x_col}: %{{x}}<br>' +
                                  f'{y_col}: %{{y:.2f}}<extra></extra>'
                ))
        
        fig.update_layout(
            xaxis_title=config.get('x_label', x_col.replace('_', ' ').title()),
            yaxis_title=config.get('y_label', 'Value'),
            barmode='group',
            bargap=0.15,
            bargroupgap=0.1
        )
        
        return self._apply_professional_styling(fig, config)
    
    def _create_scatter_chart(self, df: pd.DataFrame, config: Dict) -> go.Figure:
        """Create a professional scatter chart"""
        x_col = config.get('x_column')
        y_cols = config.get('y_columns', [])
        
        fig = go.Figure()
        colors = self._get_colors(config, len(y_cols))
        
        for idx, y_col in enumerate(y_cols):
            if y_col in df.columns:
                fig.add_trace(go.Scatter(
                    x=df[x_col],
                    y=df[y_col],
                    mode='markers',
                    name=y_col.replace('_', ' ').title(),
                    marker=dict(
                        color=colors[idx],
                        size=8,
                        line=dict(width=1, color='white'),
                        opacity=0.8
                    ),
                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                  f'{x_col}: %{{x}}<br>' +
                                  f'{y_col}: %{{y:.2f}}<extra></extra>'
                ))
        
        fig.update_layout(
            xaxis_title=config.get('x_label', x_col.replace('_', ' ').title()),
            yaxis_title=config.get('y_label', 'Value')
        )
        
        return self._apply_professional_styling(fig, config)
    
    def _create_area_chart(self, df: pd.DataFrame, config: Dict) -> go.Figure:
        """Create a professional area chart"""
        x_col = config.get('x_column')
        y_cols = config.get('y_columns', [])
        
        fig = go.Figure()
        colors = self._get_colors(config, len(y_cols))
        
        for idx, y_col in enumerate(y_cols):
            if y_col in df.columns:
                fig.add_trace(go.Scatter(
                    x=df[x_col],
                    y=df[y_col],
                    mode='lines',
                    name=y_col.replace('_', ' ').title(),
                    fill='tonexty' if idx > 0 else 'tozeroy',
                    line=dict(
                        color=colors[idx],
                        width=2
                    ),
                    fillcolor=colors[idx].replace('rgb', 'rgba').replace(')', ', 0.3)'),
                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                  f'{x_col}: %{{x}}<br>' +
                                  f'{y_col}: %{{y:.2f}}<extra></extra>'
                ))
        
        fig.update_layout(
            xaxis_title=config.get('x_label', x_col.replace('_', ' ').title()),
            yaxis_title=config.get('y_label', 'Value')
        )
        
        return self._apply_professional_styling(fig, config)
    
    def _create_pie_chart(self, df: pd.DataFrame, config: Dict) -> go.Figure:
        """Create a professional pie chart"""
        labels_col = config.get('x_column')
        values_col = config.get('y_columns', [])[0] if config.get('y_columns') else None
        
        colors = self._get_colors(config, len(df))
        
        fig = go.Figure(data=[go.Pie(
            labels=df[labels_col],
            values=df[values_col],
            hole=0.4,
            marker=dict(
                colors=colors,
                line=dict(color='white', width=2)
            ),
            textinfo='label+percent',
            textfont={'size': 12},
            hovertemplate='<b>%{label}</b><br>' +
                          'Value: %{value}<br>' +
                          'Percentage: %{percent}<extra></extra>'
        )])
        
        fig.update_layout(
            showlegend=config.get('show_legend', True),
            legend={
                'orientation': 'v',
                'yanchor': 'middle',
                'y': 0.5,
                'xanchor': 'left',
                'x': 1.05
            }
        )
        
        return self._apply_professional_styling(fig, config)
    
    def _create_heatmap(self, df: pd.DataFrame, config: Dict) -> go.Figure:
        """Create a professional heatmap"""
        # Select numeric columns
        numeric_df = df.select_dtypes(include='number')
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=[col.replace('_', ' ').title() for col in corr_matrix.columns],
            y=[col.replace('_', ' ').title() for col in corr_matrix.columns],
            colorscale='RdBu',
            zmid=0,
            zmin=-1,
            zmax=1,
            colorbar=dict(
                title='Correlation',
                titlefont={'size': 12},
                tickfont={'size': 10}
            ),
            hovertemplate='<b>%{x} vs %{y}</b><br>' +
                          'Correlation: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            xaxis=dict(
                tickangle=45,
                side='bottom'
            ),
            yaxis=dict(
                tickangle=0
            ),
            width=600,
            height=600
        )
        
        return self._apply_professional_styling(fig, config)

    def _create_stacked_bar_chart(self, df: pd.DataFrame, config: Dict) -> go.Figure:
        """Create a professional stacked bar chart"""
        x_col = config.get('x_column')
        y_cols = config.get('y_columns', [])
        
        fig = go.Figure()
        colors = self._get_colors(config, len(y_cols))
        
        for idx, y_col in enumerate(y_cols):
            if y_col in df.columns:
                fig.add_trace(go.Bar(
                    x=df[x_col],
                    y=df[y_col],
                    name=y_col.replace('_', ' ').title(),
                    marker=dict(
                        color=colors[idx],
                        line=dict(color='white', width=0.5)
                    ),
                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                  f'{x_col}: %{{x}}<br>' +
                                  f'{y_col}: %{{y:.2f}}<extra></extra>'
                ))
        
        fig.update_layout(
            xaxis_title=config.get('x_label', x_col.replace('_', ' ').title()),
            yaxis_title=config.get('y_label', 'Value'),
            barmode='stack'
        )
        
        return self._apply_professional_styling(fig, config)
    
    def _create_stacked_area_chart(self, df: pd.DataFrame, config: Dict) -> go.Figure:
        """Create a professional stacked area chart"""
        x_col = config.get('x_column')
        y_cols = config.get('y_columns', [])
        
        fig = go.Figure()
        colors = self._get_colors(config, len(y_cols))
        
        for idx, y_col in enumerate(y_cols):
            if y_col in df.columns:
                fig.add_trace(go.Scatter(
                    x=df[x_col],
                    y=df[y_col],
                    mode='lines',
                    name=y_col.replace('_', ' ').title(),
                    stackgroup='one',
                    fillcolor=colors[idx],
                    line=dict(
                        color=colors[idx],
                        width=0.5
                    ),
                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                  f'{x_col}: %{{x}}<br>' +
                                  f'{y_col}: %{{y:.2f}}<extra></extra>'
                ))
        
        fig.update_layout(
            xaxis_title=config.get('x_label', x_col.replace('_', ' ').title()),
            yaxis_title=config.get('y_label', 'Value')
        )
        
        return self._apply_professional_styling(fig, config)

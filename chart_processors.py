"""
Chart Processors - Reverse engineered from PNG outputs and test data
Each processor handles specific chart types with their data transformations
Uses Federal Reserve chart styling standards
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple
from fed_chart_style import (
    apply_fed_style, 
    FED_COLORS, 
    FED_LINE_STYLES,
    set_chart_boundaries,
    NBER_RECESSIONS
)

class PCEChartProcessor:
    """Process PCE inflation data for various chart types"""
    
    def __init__(self, data_path: str = "charting_assistant/test_pce_data.csv"):
        self.df = pd.read_csv(data_path)
        self.df['date'] = pd.to_datetime(self.df['date'])
    
    def calculate_yoy_change(self, column: str) -> pd.Series:
        """Calculate year-over-year percentage change"""
        return ((self.df[column] / self.df[column].shift(12)) - 1) * 100
    
    def calculate_mom_change(self, column: str) -> pd.Series:
        """Calculate month-over-month percentage change"""
        return ((self.df[column] / self.df[column].shift(1)) - 1) * 100
    
    def calculate_annualized_rate(self, column: str, periods: int = 3) -> pd.Series:
        """Calculate annualized rate from N-month change"""
        mom_change = self.calculate_mom_change(column)
        return ((1 + mom_change/100) ** 12 - 1) * 100
    
    def chart_01a_yoy_pce(self) -> go.Figure:
        """01a-YoY-PCE.png: Year-over-year PCE inflation"""
        fig = go.Figure()
        
        # Calculate YoY changes
        pce_yoy = self.calculate_yoy_change('pcepi')
        core_yoy = self.calculate_yoy_change('pcepicore')
        
        # Add traces with Fed colors
        fig.add_trace(go.Scatter(
            x=self.df['date'],
            y=pce_yoy,
            name='Headline PCE',
            line=dict(color=FED_COLORS['headline_inflation'], **FED_LINE_STYLES['primary']),
            mode='lines'
        ))
        
        fig.add_trace(go.Scatter(
            x=self.df['date'],
            y=core_yoy,
            name='Core PCE',
            line=dict(color=FED_COLORS['core_inflation'], **FED_LINE_STYLES['primary']),
            mode='lines'
        ))
        
        # Apply Fed styling
        fig = apply_fed_style(
            fig,
            title='PCE Inflation',
            subtitle='Year-over-year percent change',
            source='Bureau of Economic Analysis',
            add_target_line=True,
            target_value=2.0,
            add_recession_shading=True,
            recession_periods=NBER_RECESSIONS
        )
        
        fig.update_yaxes(title_text='Percent')
        fig = set_chart_boundaries(fig, 'inflation', auto_range=True)
        
        return fig

    def chart_01b_mom_pce(self) -> go.Figure:
        """01b-MoM-PCE.png: Month-over-month PCE inflation"""
        fig = go.Figure()
        
        # Calculate MoM changes
        pce_mom = self.calculate_mom_change('pcepi')
        core_mom = self.calculate_mom_change('pcepicore')
        
        fig.add_trace(go.Scatter(
            x=self.df['date'],
            y=pce_mom,
            name='PCE Inflation (MoM)',
            line=dict(color='blue', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=self.df['date'],
            y=core_mom,
            name='Core PCE Inflation (MoM)',
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            title='PCE Inflation Month-over-Month',
            xaxis_title='Date',
            yaxis_title='Percent Change (%)',
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    def chart_01c_yoy_components(self) -> go.Figure:
        """01c-YoY-components.png: YoY change in PCE components"""
        fig = go.Figure()
        
        # Calculate YoY for major components
        components = {
            'Core Goods': 'pcepi_coregoods',
            'Core Services': 'pcepi_coreservices',
            'Food': 'pcepi_food',
            'Energy': 'pcepi_energy'
        }
        
        colors = ['blue', 'green', 'orange', 'red']
        
        for (name, col), color in zip(components.items(), colors):
            yoy = self.calculate_yoy_change(col)
            fig.add_trace(go.Scatter(
                x=self.df['date'],
                y=yoy,
                name=name,
                line=dict(color=color, width=2)
            ))
        
        fig.update_layout(
            title='PCE Components Year-over-Year',
            xaxis_title='Date',
            yaxis_title='Percent Change (%)',
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    def chart_01d_mom_components(self) -> go.Figure:
        """01d-MoM-components.png: MoM change in PCE components"""
        fig = go.Figure()
        
        components = {
            'Core Goods': 'pcepi_coregoods',
            'Core Services': 'pcepi_coreservices',
            'Food': 'pcepi_food',
            'Energy': 'pcepi_energy'
        }
        
        colors = ['blue', 'green', 'orange', 'red']
        
        for (name, col), color in zip(components.items(), colors):
            mom = self.calculate_mom_change(col)
            fig.add_trace(go.Scatter(
                x=self.df['date'],
                y=mom,
                name=name,
                line=dict(color=color, width=2)
            ))
        
        fig.update_layout(
            title='PCE Components Month-over-Month',
            xaxis_title='Date',
            yaxis_title='Percent Change (%)',
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig

    def chart_02a_headline_annualized(self) -> go.Figure:
        """02a-Headline-Annualized.png: Annualized headline PCE inflation"""
        fig = go.Figure()
        
        # Calculate 1-month, 3-month, 6-month annualized rates
        pce_1m = self.calculate_annualized_rate('pcepi', 1)
        pce_3m = self.calculate_annualized_rate('pcepi', 3)
        pce_6m = self.calculate_annualized_rate('pcepi', 6)
        
        fig.add_trace(go.Scatter(
            x=self.df['date'],
            y=pce_1m,
            name='1-Month Annualized',
            line=dict(color='lightblue', width=1)
        ))
        
        fig.add_trace(go.Scatter(
            x=self.df['date'],
            y=pce_3m,
            name='3-Month Annualized',
            line=dict(color='blue', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=self.df['date'],
            y=pce_6m,
            name='6-Month Annualized',
            line=dict(color='darkblue', width=2)
        ))
        
        fig.add_hline(y=2.0, line_dash="dash", line_color="gray")
        
        fig.update_layout(
            title='Headline PCE Inflation - Annualized Rates',
            xaxis_title='Date',
            yaxis_title='Annualized Rate (%)',
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    def chart_02b_core_annualized(self) -> go.Figure:
        """02b-Core-Annualized.png: Annualized core PCE inflation"""
        fig = go.Figure()
        
        core_1m = self.calculate_annualized_rate('pcepicore', 1)
        core_3m = self.calculate_annualized_rate('pcepicore', 3)
        core_6m = self.calculate_annualized_rate('pcepicore', 6)
        
        fig.add_trace(go.Scatter(
            x=self.df['date'],
            y=core_1m,
            name='1-Month Annualized',
            line=dict(color='lightcoral', width=1)
        ))
        
        fig.add_trace(go.Scatter(
            x=self.df['date'],
            y=core_3m,
            name='3-Month Annualized',
            line=dict(color='red', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=self.df['date'],
            y=core_6m,
            name='6-Month Annualized',
            line=dict(color='darkred', width=2)
        ))
        
        fig.add_hline(y=2.0, line_dash="dash", line_color="gray")
        
        fig.update_layout(
            title='Core PCE Inflation - Annualized Rates',
            xaxis_title='Date',
            yaxis_title='Annualized Rate (%)',
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig

    def chart_3a_annualized_headlinecore(self) -> go.Figure:
        """3a-annualized-headlinecore.png: Comparison of headline vs core annualized"""
        fig = go.Figure()
        
        headline_3m = self.calculate_annualized_rate('pcepi', 3)
        core_3m = self.calculate_annualized_rate('pcepicore', 3)
        
        fig.add_trace(go.Scatter(
            x=self.df['date'],
            y=headline_3m,
            name='Headline PCE (3M Annualized)',
            line=dict(color='blue', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=self.df['date'],
            y=core_3m,
            name='Core PCE (3M Annualized)',
            line=dict(color='red', width=2)
        ))
        
        fig.add_hline(y=2.0, line_dash="dash", line_color="gray")
        
        fig.update_layout(
            title='Headline vs Core PCE - 3-Month Annualized',
            xaxis_title='Date',
            yaxis_title='Annualized Rate (%)',
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    def chart_3b_annualized_core_cats(self) -> go.Figure:
        """3b-annualized-coreCats.png: Core categories annualized"""
        fig = go.Figure()
        
        goods_3m = self.calculate_annualized_rate('pcepi_coregoods', 3)
        services_3m = self.calculate_annualized_rate('pcepi_coreservices', 3)
        
        fig.add_trace(go.Scatter(
            x=self.df['date'],
            y=goods_3m,
            name='Core Goods (3M Annualized)',
            line=dict(color='blue', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=self.df['date'],
            y=services_3m,
            name='Core Services (3M Annualized)',
            line=dict(color='green', width=2)
        ))
        
        fig.update_layout(
            title='Core PCE Categories - 3-Month Annualized',
            xaxis_title='Date',
            yaxis_title='Annualized Rate (%)',
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    def chart_3c_annualized_energy_food(self) -> go.Figure:
        """3c-annualized-energy_food.png: Energy and food annualized"""
        fig = go.Figure()
        
        energy_3m = self.calculate_annualized_rate('pcepi_energy', 3)
        food_3m = self.calculate_annualized_rate('pcepi_food', 3)
        
        fig.add_trace(go.Scatter(
            x=self.df['date'],
            y=energy_3m,
            name='Energy (3M Annualized)',
            line=dict(color='red', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=self.df['date'],
            y=food_3m,
            name='Food (3M Annualized)',
            line=dict(color='orange', width=2)
        ))
        
        fig.update_layout(
            title='Energy and Food PCE - 3-Month Annualized',
            xaxis_title='Date',
            yaxis_title='Annualized Rate (%)',
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig

    def chart_04a_headline_timing_breakdown(self) -> go.Figure:
        """04a-headline-timing-breakdown.png: Headline PCE timing breakdown"""
        fig = go.Figure()
        
        # Calculate different time horizons
        mom = self.calculate_mom_change('pcepi')
        mom_3m = mom.rolling(3).mean()
        mom_6m = mom.rolling(6).mean()
        yoy = self.calculate_yoy_change('pcepi')
        
        fig.add_trace(go.Scatter(
            x=self.df['date'],
            y=mom,
            name='Month-over-Month',
            line=dict(color='lightblue', width=1)
        ))
        
        fig.add_trace(go.Scatter(
            x=self.df['date'],
            y=mom_3m,
            name='3-Month Average',
            line=dict(color='blue', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=self.df['date'],
            y=mom_6m,
            name='6-Month Average',
            line=dict(color='darkblue', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=self.df['date'],
            y=yoy,
            name='Year-over-Year',
            line=dict(color='purple', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title='Headline PCE - Timing Breakdown',
            xaxis_title='Date',
            yaxis_title='Percent Change (%)',
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    def chart_04b_core_timing_breakdown(self) -> go.Figure:
        """04b-core-timing-breakdown.png: Core PCE timing breakdown"""
        fig = go.Figure()
        
        mom = self.calculate_mom_change('pcepicore')
        mom_3m = mom.rolling(3).mean()
        mom_6m = mom.rolling(6).mean()
        yoy = self.calculate_yoy_change('pcepicore')
        
        fig.add_trace(go.Scatter(
            x=self.df['date'],
            y=mom,
            name='Month-over-Month',
            line=dict(color='lightcoral', width=1)
        ))
        
        fig.add_trace(go.Scatter(
            x=self.df['date'],
            y=mom_3m,
            name='3-Month Average',
            line=dict(color='red', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=self.df['date'],
            y=mom_6m,
            name='6-Month Average',
            line=dict(color='darkred', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=self.df['date'],
            y=yoy,
            name='Year-over-Year',
            line=dict(color='purple', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title='Core PCE - Timing Breakdown',
            xaxis_title='Date',
            yaxis_title='Percent Change (%)',
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig

    def chart_05a_headline_decomp_yoy(self) -> go.Figure:
        """5a-headline-decomp-yoy.png: Headline PCE decomposition YoY"""
        fig = go.Figure()
        
        # Calculate contributions to headline
        food_yoy = self.calculate_yoy_change('pcepi_food')
        energy_yoy = self.calculate_yoy_change('pcepi_energy')
        core_yoy = self.calculate_yoy_change('pcepicore')
        
        # Stacked bar chart
        fig.add_trace(go.Bar(
            x=self.df['date'],
            y=food_yoy,
            name='Food',
            marker_color='orange'
        ))
        
        fig.add_trace(go.Bar(
            x=self.df['date'],
            y=energy_yoy,
            name='Energy',
            marker_color='red'
        ))
        
        fig.add_trace(go.Bar(
            x=self.df['date'],
            y=core_yoy,
            name='Core',
            marker_color='blue'
        ))
        
        fig.update_layout(
            title='Headline PCE Decomposition - Year-over-Year',
            xaxis_title='Date',
            yaxis_title='Contribution to YoY Change (%)',
            barmode='stack',
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    def chart_05b_headline_decomp_mom(self) -> go.Figure:
        """5b-headline-decomp-mom.png: Headline PCE decomposition MoM"""
        fig = go.Figure()
        
        food_mom = self.calculate_mom_change('pcepi_food')
        energy_mom = self.calculate_mom_change('pcepi_energy')
        core_mom = self.calculate_mom_change('pcepicore')
        
        fig.add_trace(go.Bar(
            x=self.df['date'],
            y=food_mom,
            name='Food',
            marker_color='orange'
        ))
        
        fig.add_trace(go.Bar(
            x=self.df['date'],
            y=energy_mom,
            name='Energy',
            marker_color='red'
        ))
        
        fig.add_trace(go.Bar(
            x=self.df['date'],
            y=core_mom,
            name='Core',
            marker_color='blue'
        ))
        
        fig.update_layout(
            title='Headline PCE Decomposition - Month-over-Month',
            xaxis_title='Date',
            yaxis_title='Contribution to MoM Change (%)',
            barmode='stack',
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig

    def chart_06a_supercore_decomp_yoy(self) -> go.Figure:
        """6a-supercore-decomp-yoy.png: Supercore PCE decomposition YoY"""
        fig = go.Figure()
        
        # Supercore = Core Services ex Housing
        supercore_yoy = self.calculate_yoy_change('pcepi_supercore')
        housing_yoy = self.calculate_yoy_change('pcepi_housing')
        goods_yoy = self.calculate_yoy_change('pcepi_coregoods')
        
        fig.add_trace(go.Bar(
            x=self.df['date'],
            y=goods_yoy,
            name='Core Goods',
            marker_color='blue'
        ))
        
        fig.add_trace(go.Bar(
            x=self.df['date'],
            y=housing_yoy,
            name='Housing Services',
            marker_color='green'
        ))
        
        fig.add_trace(go.Bar(
            x=self.df['date'],
            y=supercore_yoy,
            name='Supercore',
            marker_color='purple'
        ))
        
        fig.update_layout(
            title='Supercore PCE Decomposition - Year-over-Year',
            xaxis_title='Date',
            yaxis_title='Contribution to YoY Change (%)',
            barmode='stack',
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    def chart_06b_supercore_decomp_mom(self) -> go.Figure:
        """6b-supercore-decomp-mom.png: Supercore PCE decomposition MoM"""
        fig = go.Figure()
        
        supercore_mom = self.calculate_mom_change('pcepi_supercore')
        housing_mom = self.calculate_mom_change('pcepi_housing')
        goods_mom = self.calculate_mom_change('pcepi_coregoods')
        
        fig.add_trace(go.Bar(
            x=self.df['date'],
            y=goods_mom,
            name='Core Goods',
            marker_color='blue'
        ))
        
        fig.add_trace(go.Bar(
            x=self.df['date'],
            y=housing_mom,
            name='Housing Services',
            marker_color='green'
        ))
        
        fig.add_trace(go.Bar(
            x=self.df['date'],
            y=supercore_mom,
            name='Supercore',
            marker_color='purple'
        ))
        
        fig.update_layout(
            title='Supercore PCE Decomposition - Month-over-Month',
            xaxis_title='Date',
            yaxis_title='Contribution to MoM Change (%)',
            barmode='stack',
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig


class SupplyDemandChartProcessor:
    """Process supply/demand driven inflation data"""
    
    def __init__(self, data_path: str = "charting_assistant/test_supply_demand_data.csv"):
        self.df = pd.read_csv(data_path)
        self.df['date'] = pd.to_datetime(self.df['date'])
    
    def chart_fig7a_supply_demand_headline(self) -> go.Figure:
        """fig7a.png: Supply vs Demand driven inflation (headline)"""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=self.df['date'],
            y=self.df['Demand-driven Inflation (headline, y/y)'],
            name='Demand-driven',
            line=dict(color='blue', width=2),
            stackgroup='one'
        ))
        
        fig.add_trace(go.Scatter(
            x=self.df['date'],
            y=self.df['Supply-driven Inflation (headline, y/y)'],
            name='Supply-driven',
            line=dict(color='red', width=2),
            stackgroup='one'
        ))
        
        fig.add_trace(go.Scatter(
            x=self.df['date'],
            y=self.df['Ambiguous (headline, y/y)'],
            name='Ambiguous',
            line=dict(color='gray', width=2),
            stackgroup='one'
        ))
        
        fig.update_layout(
            title='Supply vs Demand Driven Inflation (Headline)',
            xaxis_title='Date',
            yaxis_title='Contribution to YoY Inflation (%)',
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    def chart_fig7b_supply_demand_core(self) -> go.Figure:
        """fig7b.png: Supply vs Demand driven inflation (core)"""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=self.df['date'],
            y=self.df['Demand-driven Inflation (core, y/y)'],
            name='Demand-driven',
            line=dict(color='blue', width=2),
            stackgroup='one'
        ))
        
        fig.add_trace(go.Scatter(
            x=self.df['date'],
            y=self.df['Supply-driven Inflation (core, y/y)'],
            name='Supply-driven',
            line=dict(color='red', width=2),
            stackgroup='one'
        ))
        
        fig.add_trace(go.Scatter(
            x=self.df['date'],
            y=self.df['Ambiguous (core, y/y)'],
            name='Ambiguous',
            line=dict(color='gray', width=2),
            stackgroup='one'
        ))
        
        fig.update_layout(
            title='Supply vs Demand Driven Inflation (Core)',
            xaxis_title='Date',
            yaxis_title='Contribution to YoY Inflation (%)',
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig


# Chart registry for easy access
CHART_REGISTRY = {
    '01a-YoY-PCE': ('PCEChartProcessor', 'chart_01a_yoy_pce'),
    '01b-MoM-PCE': ('PCEChartProcessor', 'chart_01b_mom_pce'),
    '01c-YoY-components': ('PCEChartProcessor', 'chart_01c_yoy_components'),
    '01d-MoM-components': ('PCEChartProcessor', 'chart_01d_mom_components'),
    '02a-Headline-Annualized': ('PCEChartProcessor', 'chart_02a_headline_annualized'),
    '02b-Core-Annualized': ('PCEChartProcessor', 'chart_02b_core_annualized'),
    '3a-annualized-headlinecore': ('PCEChartProcessor', 'chart_3a_annualized_headlinecore'),
    '3b-annualized-coreCats': ('PCEChartProcessor', 'chart_3b_annualized_core_cats'),
    '3c-annualized-energy_food': ('PCEChartProcessor', 'chart_3c_annualized_energy_food'),
    '04a-headline-timing-breakdown': ('PCEChartProcessor', 'chart_04a_headline_timing_breakdown'),
    '04b-core-timing-breakdown': ('PCEChartProcessor', 'chart_04b_core_timing_breakdown'),
    '5a-headline-decomp-yoy': ('PCEChartProcessor', 'chart_05a_headline_decomp_yoy'),
    '5b-headline-decomp-mom': ('PCEChartProcessor', 'chart_05b_headline_decomp_mom'),
    '6a-supercore-decomp-yoy': ('PCEChartProcessor', 'chart_06a_supercore_decomp_yoy'),
    '6b-supercore-decomp-mom': ('PCEChartProcessor', 'chart_06b_supercore_decomp_mom'),
    'fig7a': ('SupplyDemandChartProcessor', 'chart_fig7a_supply_demand_headline'),
    'fig7b': ('SupplyDemandChartProcessor', 'chart_fig7b_supply_demand_core'),
}


def generate_chart(chart_name: str) -> go.Figure:
    """Generate a chart by name"""
    if chart_name not in CHART_REGISTRY:
        raise ValueError(f"Unknown chart: {chart_name}. Available: {list(CHART_REGISTRY.keys())}")
    
    processor_class, method_name = CHART_REGISTRY[chart_name]
    
    if processor_class == 'PCEChartProcessor':
        processor = PCEChartProcessor()
    elif processor_class == 'SupplyDemandChartProcessor':
        processor = SupplyDemandChartProcessor()
    else:
        raise ValueError(f"Unknown processor: {processor_class}")
    
    method = getattr(processor, method_name)
    return method()

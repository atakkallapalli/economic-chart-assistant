"""
Complete Data Processors for All Chart Types
Reverse engineered from PNG outputs and test data files
Each processor includes data transformation logic specific to the chart
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional
from fed_chart_style import (
    apply_fed_style, 
    FED_COLORS, 
    FED_LINE_STYLES,
    set_chart_boundaries,
    NBER_RECESSIONS,
    get_color_scheme
)


class BaseChartProcessor:
    """Base class for all chart processors"""
    
    def __init__(self, pce_data_path: str = "charting_assistant/test_pce_data.csv",
                 supply_demand_path: str = "charting_assistant/test_supply_demand_data.csv"):
        self.pce_df = pd.read_csv(pce_data_path)
        self.pce_df['date'] = pd.to_datetime(self.pce_df['date'])
        
        try:
            self.sd_df = pd.read_csv(supply_demand_path)
            self.sd_df['date'] = pd.to_datetime(self.sd_df['date'])
        except:
            self.sd_df = None
    
    def calculate_yoy_change(self, column: str, df: Optional[pd.DataFrame] = None) -> pd.Series:
        """Calculate year-over-year percentage change"""
        if df is None:
            df = self.pce_df
        return ((df[column] / df[column].shift(12)) - 1) * 100
    
    def calculate_mom_change(self, column: str, df: Optional[pd.DataFrame] = None) -> pd.Series:
        """Calculate month-over-month percentage change"""
        if df is None:
            df = self.pce_df
        return ((df[column] / df[column].shift(1)) - 1) * 100
    
    def calculate_annualized_rate(self, column: str, periods: int = 3, 
                                  df: Optional[pd.DataFrame] = None) -> pd.Series:
        """Calculate annualized rate from N-month change"""
        if df is None:
            df = self.pce_df
        
        # Calculate N-period change
        n_period_change = (df[column] / df[column].shift(periods)) - 1
        # Annualize: (1 + change)^(12/N) - 1
        annualized = ((1 + n_period_change) ** (12 / periods) - 1) * 100
        return annualized
    
    def calculate_contribution(self, component_col: str, total_col: str,
                              df: Optional[pd.DataFrame] = None) -> pd.Series:
        """Calculate component contribution to total change"""
        if df is None:
            df = self.pce_df
        
        # Weight of component in total
        weight = df[component_col] / df[total_col]
        # Component change
        comp_change = self.calculate_mom_change(component_col, df)
        # Contribution = weight * change
        return weight.shift(1) * comp_change


class Chart01Processor(BaseChartProcessor):
    """Processors for Chart 01 series (YoY and MoM PCE)"""
    
    def chart_01a_yoy_pce(self) -> go.Figure:
        """01a-YoY-PCE.png: Year-over-year PCE inflation"""
        fig = go.Figure()
        
        # Calculate YoY changes
        pce_yoy = self.calculate_yoy_change('pcepi')
        core_yoy = self.calculate_yoy_change('pcepicore')
        
        # Add traces with Fed colors
        fig.add_trace(go.Scatter(
            x=self.pce_df['date'],
            y=pce_yoy,
            name='Headline PCE',
            line=dict(color=FED_COLORS['headline_inflation'], **FED_LINE_STYLES['primary']),
            mode='lines'
        ))
        
        fig.add_trace(go.Scatter(
            x=self.pce_df['date'],
            y=core_yoy,
            name='Core PCE',
            line=dict(color=FED_COLORS['core_inflation'], **FED_LINE_STYLES['primary']),
            mode='lines'
        ))
        
        # Apply Fed styling
        fig = apply_fed_style(
            fig,
            title='Personal Consumption Expenditures Price Index',
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
            x=self.pce_df['date'],
            y=pce_mom,
            name='Headline PCE',
            line=dict(color=FED_COLORS['headline_inflation'], **FED_LINE_STYLES['primary']),
            mode='lines'
        ))
        
        fig.add_trace(go.Scatter(
            x=self.pce_df['date'],
            y=core_mom,
            name='Core PCE',
            line=dict(color=FED_COLORS['core_inflation'], **FED_LINE_STYLES['primary']),
            mode='lines'
        ))
        
        fig = apply_fed_style(
            fig,
            title='Personal Consumption Expenditures Price Index',
            subtitle='Month-over-month percent change',
            source='Bureau of Economic Analysis',
            add_recession_shading=True,
            recession_periods=NBER_RECESSIONS
        )
        
        fig.update_yaxes(title_text='Percent')
        fig = set_chart_boundaries(fig, 'inflation_mom', auto_range=True)
        
        return fig
    
    def chart_01c_yoy_components(self) -> go.Figure:
        """01c-YoY-components.png: YoY change in PCE components"""
        fig = go.Figure()
        
        # Calculate YoY for major components
        components = {
            'Core Goods': ('pcepi_coregoods', FED_COLORS['goods']),
            'Core Services': ('pcepi_coreservices', FED_COLORS['services']),
            'Food': ('pcepi_food', FED_COLORS['food']),
            'Energy': ('pcepi_energy', FED_COLORS['energy'])
        }
        
        for name, (col, color) in components.items():
            yoy = self.calculate_yoy_change(col)
            fig.add_trace(go.Scatter(
                x=self.pce_df['date'],
                y=yoy,
                name=name,
                line=dict(color=color, **FED_LINE_STYLES['primary']),
                mode='lines'
            ))
        
        fig = apply_fed_style(
            fig,
            title='PCE Components',
            subtitle='Year-over-year percent change',
            source='Bureau of Economic Analysis',
            add_recession_shading=True,
            recession_periods=NBER_RECESSIONS
        )
        
        fig.update_yaxes(title_text='Percent')
        
        return fig
    
    def chart_01d_mom_components(self) -> go.Figure:
        """01d-MoM-components.png: MoM change in PCE components"""
        fig = go.Figure()
        
        components = {
            'Core Goods': ('pcepi_coregoods', FED_COLORS['goods']),
            'Core Services': ('pcepi_coreservices', FED_COLORS['services']),
            'Food': ('pcepi_food', FED_COLORS['food']),
            'Energy': ('pcepi_energy', FED_COLORS['energy'])
        }
        
        for name, (col, color) in components.items():
            mom = self.calculate_mom_change(col)
            fig.add_trace(go.Scatter(
                x=self.pce_df['date'],
                y=mom,
                name=name,
                line=dict(color=color, **FED_LINE_STYLES['primary']),
                mode='lines'
            ))
        
        fig = apply_fed_style(
            fig,
            title='PCE Components',
            subtitle='Month-over-month percent change',
            source='Bureau of Economic Analysis',
            add_recession_shading=True,
            recession_periods=NBER_RECESSIONS
        )
        
        fig.update_yaxes(title_text='Percent')
        
        return fig


class Chart02Processor(BaseChartProcessor):
    """Processors for Chart 02 series (Annualized rates)"""
    
    def chart_02a_headline_annualized(self) -> go.Figure:
        """02a-Headline-Annualized.png: Annualized headline PCE inflation"""
        fig = go.Figure()
        
        # Calculate 1-month, 3-month, 6-month annualized rates
        pce_1m = self.calculate_annualized_rate('pcepi', 1)
        pce_3m = self.calculate_annualized_rate('pcepi', 3)
        pce_6m = self.calculate_annualized_rate('pcepi', 6)
        
        fig.add_trace(go.Scatter(
            x=self.pce_df['date'],
            y=pce_1m,
            name='1-Month',
            line=dict(color=FED_COLORS['primary_blue'], width=1.5, dash='dot'),
            mode='lines',
            opacity=0.6
        ))
        
        fig.add_trace(go.Scatter(
            x=self.pce_df['date'],
            y=pce_3m,
            name='3-Month',
            line=dict(color=FED_COLORS['headline_inflation'], **FED_LINE_STYLES['primary']),
            mode='lines'
        ))
        
        fig.add_trace(go.Scatter(
            x=self.pce_df['date'],
            y=pce_6m,
            name='6-Month',
            line=dict(color=FED_COLORS['secondary_teal'], **FED_LINE_STYLES['secondary']),
            mode='lines'
        ))
        
        fig = apply_fed_style(
            fig,
            title='Headline PCE Inflation',
            subtitle='Annualized rates',
            source='Bureau of Economic Analysis',
            add_target_line=True,
            target_value=2.0,
            add_recession_shading=True,
            recession_periods=NBER_RECESSIONS
        )
        
        fig.update_yaxes(title_text='Percent, annualized')
        fig = set_chart_boundaries(fig, 'inflation', auto_range=True)
        
        return fig
    
    def chart_02b_core_annualized(self) -> go.Figure:
        """02b-Core-Annualized.png: Annualized core PCE inflation"""
        fig = go.Figure()
        
        core_1m = self.calculate_annualized_rate('pcepicore', 1)
        core_3m = self.calculate_annualized_rate('pcepicore', 3)
        core_6m = self.calculate_annualized_rate('pcepicore', 6)
        
        fig.add_trace(go.Scatter(
            x=self.pce_df['date'],
            y=core_1m,
            name='1-Month',
            line=dict(color=FED_COLORS['primary_red'], width=1.5, dash='dot'),
            mode='lines',
            opacity=0.6
        ))
        
        fig.add_trace(go.Scatter(
            x=self.pce_df['date'],
            y=core_3m,
            name='3-Month',
            line=dict(color=FED_COLORS['core_inflation'], **FED_LINE_STYLES['primary']),
            mode='lines'
        ))
        
        fig.add_trace(go.Scatter(
            x=self.pce_df['date'],
            y=core_6m,
            name='6-Month',
            line=dict(color=FED_COLORS['secondary_purple'], **FED_LINE_STYLES['secondary']),
            mode='lines'
        ))
        
        fig = apply_fed_style(
            fig,
            title='Core PCE Inflation',
            subtitle='Annualized rates',
            source='Bureau of Economic Analysis',
            add_target_line=True,
            target_value=2.0,
            add_recession_shading=True,
            recession_periods=NBER_RECESSIONS
        )
        
        fig.update_yaxes(title_text='Percent, annualized')
        fig = set_chart_boundaries(fig, 'inflation', auto_range=True)
        
        return fig


class Chart03Processor(BaseChartProcessor):
    """Processors for Chart 03 series (Annualized comparisons)"""
    
    def chart_3a_annualized_headlinecore(self) -> go.Figure:
        """3a-annualized-headlinecore.png: Comparison of headline vs core annualized"""
        fig = go.Figure()
        
        headline_3m = self.calculate_annualized_rate('pcepi', 3)
        core_3m = self.calculate_annualized_rate('pcepicore', 3)
        
        fig.add_trace(go.Scatter(
            x=self.pce_df['date'],
            y=headline_3m,
            name='Headline PCE',
            line=dict(color=FED_COLORS['headline_inflation'], **FED_LINE_STYLES['primary']),
            mode='lines'
        ))
        
        fig.add_trace(go.Scatter(
            x=self.pce_df['date'],
            y=core_3m,
            name='Core PCE',
            line=dict(color=FED_COLORS['core_inflation'], **FED_LINE_STYLES['primary']),
            mode='lines'
        ))
        
        fig = apply_fed_style(
            fig,
            title='Headline vs Core PCE Inflation',
            subtitle='3-month annualized rate',
            source='Bureau of Economic Analysis',
            add_target_line=True,
            target_value=2.0,
            add_recession_shading=True,
            recession_periods=NBER_RECESSIONS
        )
        
        fig.update_yaxes(title_text='Percent, annualized')
        fig = set_chart_boundaries(fig, 'inflation', auto_range=True)
        
        return fig
    
    def chart_3b_annualized_core_cats(self) -> go.Figure:
        """3b-annualized-coreCats.png: Core categories annualized"""
        fig = go.Figure()
        
        goods_3m = self.calculate_annualized_rate('pcepi_coregoods', 3)
        services_3m = self.calculate_annualized_rate('pcepi_coreservices', 3)
        
        fig.add_trace(go.Scatter(
            x=self.pce_df['date'],
            y=goods_3m,
            name='Core Goods',
            line=dict(color=FED_COLORS['goods'], **FED_LINE_STYLES['primary']),
            mode='lines'
        ))
        
        fig.add_trace(go.Scatter(
            x=self.pce_df['date'],
            y=services_3m,
            name='Core Services',
            line=dict(color=FED_COLORS['services'], **FED_LINE_STYLES['primary']),
            mode='lines'
        ))
        
        fig = apply_fed_style(
            fig,
            title='Core PCE Categories',
            subtitle='3-month annualized rate',
            source='Bureau of Economic Analysis',
            add_recession_shading=True,
            recession_periods=NBER_RECESSIONS
        )
        
        fig.update_yaxes(title_text='Percent, annualized')
        
        return fig
    
    def chart_3c_annualized_energy_food(self) -> go.Figure:
        """3c-annualized-energy_food.png: Energy and food annualized"""
        fig = go.Figure()
        
        energy_3m = self.calculate_annualized_rate('pcepi_energy', 3)
        food_3m = self.calculate_annualized_rate('pcepi_food', 3)
        
        fig.add_trace(go.Scatter(
            x=self.pce_df['date'],
            y=energy_3m,
            name='Energy',
            line=dict(color=FED_COLORS['energy'], **FED_LINE_STYLES['primary']),
            mode='lines'
        ))
        
        fig.add_trace(go.Scatter(
            x=self.pce_df['date'],
            y=food_3m,
            name='Food',
            line=dict(color=FED_COLORS['food'], **FED_LINE_STYLES['primary']),
            mode='lines'
        ))
        
        fig = apply_fed_style(
            fig,
            title='Energy and Food PCE',
            subtitle='3-month annualized rate',
            source='Bureau of Economic Analysis',
            add_recession_shading=True,
            recession_periods=NBER_RECESSIONS
        )
        
        fig.update_yaxes(title_text='Percent, annualized')
        
        return fig


class Chart04Processor(BaseChartProcessor):
    """Processors for Chart 04 series (Timing breakdown)"""
    
    def chart_04a_headline_timing_breakdown(self) -> go.Figure:
        """04a-headline-timing-breakdown.png: Headline PCE timing breakdown"""
        fig = go.Figure()
        
        # Calculate different time horizons
        mom = self.calculate_mom_change('pcepi')
        mom_3m = mom.rolling(3).mean()
        mom_6m = mom.rolling(6).mean()
        yoy = self.calculate_yoy_change('pcepi')
        
        fig.add_trace(go.Scatter(
            x=self.pce_df['date'],
            y=mom,
            name='Month-over-Month',
            line=dict(color=FED_COLORS['primary_blue'], width=1, dash='dot'),
            mode='lines',
            opacity=0.5
        ))
        
        fig.add_trace(go.Scatter(
            x=self.pce_df['date'],
            y=mom_3m,
            name='3-Month Average',
            line=dict(color=FED_COLORS['headline_inflation'], **FED_LINE_STYLES['primary']),
            mode='lines'
        ))
        
        fig.add_trace(go.Scatter(
            x=self.pce_df['date'],
            y=mom_6m,
            name='6-Month Average',
            line=dict(color=FED_COLORS['secondary_teal'], **FED_LINE_STYLES['secondary']),
            mode='lines'
        ))
        
        fig.add_trace(go.Scatter(
            x=self.pce_df['date'],
            y=yoy,
            name='Year-over-Year',
            line=dict(color=FED_COLORS['primary_purple'], width=2, dash='dash'),
            mode='lines'
        ))
        
        fig = apply_fed_style(
            fig,
            title='Headline PCE Inflation',
            subtitle='Multiple time horizons',
            source='Bureau of Economic Analysis',
            add_recession_shading=True,
            recession_periods=NBER_RECESSIONS
        )
        
        fig.update_yaxes(title_text='Percent change')
        
        return fig
    
    def chart_04b_core_timing_breakdown(self) -> go.Figure:
        """04b-core-timing-breakdown.png: Core PCE timing breakdown"""
        fig = go.Figure()
        
        mom = self.calculate_mom_change('pcepicore')
        mom_3m = mom.rolling(3).mean()
        mom_6m = mom.rolling(6).mean()
        yoy = self.calculate_yoy_change('pcepicore')
        
        fig.add_trace(go.Scatter(
            x=self.pce_df['date'],
            y=mom,
            name='Month-over-Month',
            line=dict(color=FED_COLORS['primary_red'], width=1, dash='dot'),
            mode='lines',
            opacity=0.5
        ))
        
        fig.add_trace(go.Scatter(
            x=self.pce_df['date'],
            y=mom_3m,
            name='3-Month Average',
            line=dict(color=FED_COLORS['core_inflation'], **FED_LINE_STYLES['primary']),
            mode='lines'
        ))
        
        fig.add_trace(go.Scatter(
            x=self.pce_df['date'],
            y=mom_6m,
            name='6-Month Average',
            line=dict(color=FED_COLORS['secondary_purple'], **FED_LINE_STYLES['secondary']),
            mode='lines'
        ))
        
        fig.add_trace(go.Scatter(
            x=self.pce_df['date'],
            y=yoy,
            name='Year-over-Year',
            line=dict(color=FED_COLORS['primary_purple'], width=2, dash='dash'),
            mode='lines'
        ))
        
        fig = apply_fed_style(
            fig,
            title='Core PCE Inflation',
            subtitle='Multiple time horizons',
            source='Bureau of Economic Analysis',
            add_recession_shading=True,
            recession_periods=NBER_RECESSIONS
        )
        
        fig.update_yaxes(title_text='Percent change')
        
        return fig



class Chart05Processor(BaseChartProcessor):
    """Processors for Chart 05 series (Headline decomposition)"""
    
    def chart_05a_headline_decomp_yoy(self) -> go.Figure:
        """5a-headline-decomp-yoy.png: Headline PCE decomposition YoY"""
        fig = go.Figure()
        
        # Calculate YoY contributions
        # Weights from PCE spending shares
        food_weight = self.pce_df['pce_food'] / self.pce_df['pce']
        energy_weight = self.pce_df['pce_energy'] / self.pce_df['pce']
        core_weight = 1 - food_weight - energy_weight
        
        # YoY changes
        food_yoy = self.calculate_yoy_change('pcepi_food')
        energy_yoy = self.calculate_yoy_change('pcepi_energy')
        core_yoy = self.calculate_yoy_change('pcepicore')
        
        # Contributions = weight * change
        food_contrib = food_weight.shift(12) * food_yoy / 100
        energy_contrib = energy_weight.shift(12) * energy_yoy / 100
        core_contrib = core_weight.shift(12) * core_yoy / 100
        
        # Stacked bar chart
        fig.add_trace(go.Bar(
            x=self.pce_df['date'],
            y=core_contrib,
            name='Core',
            marker_color=FED_COLORS['goods']
        ))
        
        fig.add_trace(go.Bar(
            x=self.pce_df['date'],
            y=food_contrib,
            name='Food',
            marker_color=FED_COLORS['food']
        ))
        
        fig.add_trace(go.Bar(
            x=self.pce_df['date'],
            y=energy_contrib,
            name='Energy',
            marker_color=FED_COLORS['energy']
        ))
        
        fig.update_layout(barmode='stack')
        
        fig = apply_fed_style(
            fig,
            title='Headline PCE Decomposition',
            subtitle='Contribution to year-over-year change',
            source='Bureau of Economic Analysis',
            add_recession_shading=True,
            recession_periods=NBER_RECESSIONS
        )
        
        fig.update_yaxes(title_text='Percentage points')
        
        return fig
    
    def chart_05b_headline_decomp_mom(self) -> go.Figure:
        """5b-headline-decomp-mom.png: Headline PCE decomposition MoM"""
        fig = go.Figure()
        
        # Calculate MoM contributions
        food_weight = self.pce_df['pce_food'] / self.pce_df['pce']
        energy_weight = self.pce_df['pce_energy'] / self.pce_df['pce']
        core_weight = 1 - food_weight - energy_weight
        
        food_mom = self.calculate_mom_change('pcepi_food')
        energy_mom = self.calculate_mom_change('pcepi_energy')
        core_mom = self.calculate_mom_change('pcepicore')
        
        food_contrib = food_weight.shift(1) * food_mom / 100
        energy_contrib = energy_weight.shift(1) * energy_mom / 100
        core_contrib = core_weight.shift(1) * core_mom / 100
        
        fig.add_trace(go.Bar(
            x=self.pce_df['date'],
            y=core_contrib,
            name='Core',
            marker_color=FED_COLORS['goods']
        ))
        
        fig.add_trace(go.Bar(
            x=self.pce_df['date'],
            y=food_contrib,
            name='Food',
            marker_color=FED_COLORS['food']
        ))
        
        fig.add_trace(go.Bar(
            x=self.pce_df['date'],
            y=energy_contrib,
            name='Energy',
            marker_color=FED_COLORS['energy']
        ))
        
        fig.update_layout(barmode='stack')
        
        fig = apply_fed_style(
            fig,
            title='Headline PCE Decomposition',
            subtitle='Contribution to month-over-month change',
            source='Bureau of Economic Analysis'
        )
        
        fig.update_yaxes(title_text='Percentage points')
        
        return fig


class Chart06Processor(BaseChartProcessor):
    """Processors for Chart 06 series (Supercore decomposition)"""
    
    def chart_06a_supercore_decomp_yoy(self) -> go.Figure:
        """6a-supercore-decomp-yoy.png: Supercore PCE decomposition YoY"""
        fig = go.Figure()
        
        # Supercore = Core Services ex Housing
        # Calculate contributions
        goods_weight = self.pce_df['pce_coregoods'] / self.pce_df['pce_core']
        housing_weight = self.pce_df['pce_housing'] / self.pce_df['pce_core']
        supercore_weight = self.pce_df['pce_supercore'] / self.pce_df['pce_core']
        
        goods_yoy = self.calculate_yoy_change('pcepi_coregoods')
        housing_yoy = self.calculate_yoy_change('pcepi_housing')
        supercore_yoy = self.calculate_yoy_change('pcepi_supercore')
        
        goods_contrib = goods_weight.shift(12) * goods_yoy / 100
        housing_contrib = housing_weight.shift(12) * housing_yoy / 100
        supercore_contrib = supercore_weight.shift(12) * supercore_yoy / 100
        
        fig.add_trace(go.Bar(
            x=self.pce_df['date'],
            y=goods_contrib,
            name='Core Goods',
            marker_color=FED_COLORS['goods']
        ))
        
        fig.add_trace(go.Bar(
            x=self.pce_df['date'],
            y=housing_contrib,
            name='Housing Services',
            marker_color=FED_COLORS['housing']
        ))
        
        fig.add_trace(go.Bar(
            x=self.pce_df['date'],
            y=supercore_contrib,
            name='Supercore',
            marker_color=FED_COLORS['supercore']
        ))
        
        fig.update_layout(barmode='stack')
        
        fig = apply_fed_style(
            fig,
            title='Core PCE Decomposition',
            subtitle='Contribution to year-over-year change',
            source='Bureau of Economic Analysis',
            add_recession_shading=True,
            recession_periods=NBER_RECESSIONS
        )
        
        fig.update_yaxes(title_text='Percentage points')
        
        return fig
    
    def chart_06b_supercore_decomp_mom(self) -> go.Figure:
        """6b-supercore-decomp-mom.png: Supercore PCE decomposition MoM"""
        fig = go.Figure()
        
        goods_weight = self.pce_df['pce_coregoods'] / self.pce_df['pce_core']
        housing_weight = self.pce_df['pce_housing'] / self.pce_df['pce_core']
        supercore_weight = self.pce_df['pce_supercore'] / self.pce_df['pce_core']
        
        goods_mom = self.calculate_mom_change('pcepi_coregoods')
        housing_mom = self.calculate_mom_change('pcepi_housing')
        supercore_mom = self.calculate_mom_change('pcepi_supercore')
        
        goods_contrib = goods_weight.shift(1) * goods_mom / 100
        housing_contrib = housing_weight.shift(1) * housing_mom / 100
        supercore_contrib = supercore_weight.shift(1) * supercore_mom / 100
        
        fig.add_trace(go.Bar(
            x=self.pce_df['date'],
            y=goods_contrib,
            name='Core Goods',
            marker_color=FED_COLORS['goods']
        ))
        
        fig.add_trace(go.Bar(
            x=self.pce_df['date'],
            y=housing_contrib,
            name='Housing Services',
            marker_color=FED_COLORS['housing']
        ))
        
        fig.add_trace(go.Bar(
            x=self.pce_df['date'],
            y=supercore_contrib,
            name='Supercore',
            marker_color=FED_COLORS['supercore']
        ))
        
        fig.update_layout(barmode='stack')
        
        fig = apply_fed_style(
            fig,
            title='Core PCE Decomposition',
            subtitle='Contribution to month-over-month change',
            source='Bureau of Economic Analysis'
        )
        
        fig.update_yaxes(title_text='Percentage points')
        
        return fig


class Chart07Processor(BaseChartProcessor):
    """Processors for Chart 07 series (Supply/Demand decomposition)"""
    
    def chart_fig7a_supply_demand_headline(self) -> go.Figure:
        """fig7a.png: Supply vs Demand driven inflation (headline)"""
        if self.sd_df is None:
            raise ValueError("Supply/demand data not available")
        
        fig = go.Figure()
        
        # Area chart with stacking
        fig.add_trace(go.Scatter(
            x=self.sd_df['date'],
            y=self.sd_df['Demand-driven Inflation (headline, y/y)'],
            name='Demand-driven',
            line=dict(color=FED_COLORS['primary_blue'], width=0),
            fillcolor='rgba(68, 114, 196, 0.6)',
            fill='tonexty',
            stackgroup='one',
            mode='lines'
        ))
        
        fig.add_trace(go.Scatter(
            x=self.sd_df['date'],
            y=self.sd_df['Supply-driven Inflation (headline, y/y)'],
            name='Supply-driven',
            line=dict(color=FED_COLORS['primary_red'], width=0),
            fillcolor='rgba(197, 80, 75, 0.6)',
            fill='tonexty',
            stackgroup='one',
            mode='lines'
        ))
        
        fig.add_trace(go.Scatter(
            x=self.sd_df['date'],
            y=self.sd_df['Ambiguous (headline, y/y)'],
            name='Ambiguous',
            line=dict(color=FED_COLORS['medium_gray'], width=0),
            fillcolor='rgba(128, 128, 128, 0.4)',
            fill='tonexty',
            stackgroup='one',
            mode='lines'
        ))
        
        fig = apply_fed_style(
            fig,
            title='Supply vs Demand Driven Inflation',
            subtitle='Headline PCE, year-over-year',
            source='Shapiro (2022) methodology, BEA data',
            add_recession_shading=True,
            recession_periods=NBER_RECESSIONS
        )
        
        fig.update_yaxes(title_text='Percentage points')
        
        return fig
    
    def chart_fig7b_supply_demand_core(self) -> go.Figure:
        """fig7b.png: Supply vs Demand driven inflation (core)"""
        if self.sd_df is None:
            raise ValueError("Supply/demand data not available")
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=self.sd_df['date'],
            y=self.sd_df['Demand-driven Inflation (core, y/y)'],
            name='Demand-driven',
            line=dict(color=FED_COLORS['primary_blue'], width=0),
            fillcolor='rgba(68, 114, 196, 0.6)',
            fill='tonexty',
            stackgroup='one',
            mode='lines'
        ))
        
        fig.add_trace(go.Scatter(
            x=self.sd_df['date'],
            y=self.sd_df['Supply-driven Inflation (core, y/y)'],
            name='Supply-driven',
            line=dict(color=FED_COLORS['primary_red'], width=0),
            fillcolor='rgba(197, 80, 75, 0.6)',
            fill='tonexty',
            stackgroup='one',
            mode='lines'
        ))
        
        fig.add_trace(go.Scatter(
            x=self.sd_df['date'],
            y=self.sd_df['Ambiguous (core, y/y)'],
            name='Ambiguous',
            line=dict(color=FED_COLORS['medium_gray'], width=0),
            fillcolor='rgba(128, 128, 128, 0.4)',
            fill='tonexty',
            stackgroup='one',
            mode='lines'
        ))
        
        fig = apply_fed_style(
            fig,
            title='Supply vs Demand Driven Inflation',
            subtitle='Core PCE, year-over-year',
            source='Shapiro (2022) methodology, BEA data',
            add_recession_shading=True,
            recession_periods=NBER_RECESSIONS
        )
        
        fig.update_yaxes(title_text='Percentage points')
        
        return fig


# Master chart registry with all processors
CHART_REGISTRY = {
    # Chart 01 series - Basic YoY and MoM
    '01a-YoY-PCE': (Chart01Processor, 'chart_01a_yoy_pce'),
    '01b-MoM-PCE': (Chart01Processor, 'chart_01b_mom_pce'),
    '01c-YoY-components': (Chart01Processor, 'chart_01c_yoy_components'),
    '01d-MoM-components': (Chart01Processor, 'chart_01d_mom_components'),
    
    # Chart 02 series - Annualized rates
    '02a-Headline-Annualized': (Chart02Processor, 'chart_02a_headline_annualized'),
    '02b-Core-Annualized': (Chart02Processor, 'chart_02b_core_annualized'),
    
    # Chart 03 series - Annualized comparisons
    '3a-annualized-headlinecore': (Chart03Processor, 'chart_3a_annualized_headlinecore'),
    '3b-annualized-coreCats': (Chart03Processor, 'chart_3b_annualized_core_cats'),
    '3c-annualized-energy_food': (Chart03Processor, 'chart_3c_annualized_energy_food'),
    
    # Chart 04 series - Timing breakdown
    '04a-headline-timing-breakdown': (Chart04Processor, 'chart_04a_headline_timing_breakdown'),
    '04b-core-timing-breakdown': (Chart04Processor, 'chart_04b_core_timing_breakdown'),
    
    # Chart 05 series - Headline decomposition
    '5a-headline-decomp-yoy': (Chart05Processor, 'chart_05a_headline_decomp_yoy'),
    '5b-headline-decomp-mom': (Chart05Processor, 'chart_05b_headline_decomp_mom'),
    
    # Chart 06 series - Supercore decomposition
    '6a-supercore-decomp-yoy': (Chart06Processor, 'chart_06a_supercore_decomp_yoy'),
    '6b-supercore-decomp-mom': (Chart06Processor, 'chart_06b_supercore_decomp_mom'),
    
    # Chart 07 series - Supply/Demand
    'fig7a': (Chart07Processor, 'chart_fig7a_supply_demand_headline'),
    'fig7b': (Chart07Processor, 'chart_fig7b_supply_demand_core'),
}


def generate_chart(chart_name: str, 
                  pce_data_path: str = "charting_assistant/test_pce_data.csv",
                  supply_demand_path: str = "charting_assistant/test_supply_demand_data.csv") -> go.Figure:
    """
    Generate a chart by name
    
    Args:
        chart_name: Name of the chart (e.g., '01a-YoY-PCE')
        pce_data_path: Path to PCE data CSV
        supply_demand_path: Path to supply/demand data CSV
    
    Returns:
        Plotly Figure object
    
    Raises:
        ValueError: If chart name is not recognized
    """
    if chart_name not in CHART_REGISTRY:
        available = '\n  - '.join(sorted(CHART_REGISTRY.keys()))
        raise ValueError(
            f"Unknown chart: {chart_name}\n\n"
            f"Available charts:\n  - {available}"
        )
    
    processor_class, method_name = CHART_REGISTRY[chart_name]
    processor = processor_class(pce_data_path, supply_demand_path)
    method = getattr(processor, method_name)
    
    return method()


def generate_all_charts(output_dir: str = "output_charts",
                       pce_data_path: str = "charting_assistant/test_pce_data.csv",
                       supply_demand_path: str = "charting_assistant/test_supply_demand_data.csv"):
    """
    Generate all charts and save to output directory
    
    Args:
        output_dir: Directory to save charts
        pce_data_path: Path to PCE data CSV
        supply_demand_path: Path to supply/demand data CSV
    """
    import os
    from pathlib import Path
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {len(CHART_REGISTRY)} charts...")
    print()
    
    for chart_name in sorted(CHART_REGISTRY.keys()):
        try:
            print(f"  Generating {chart_name}...", end=' ')
            fig = generate_chart(chart_name, pce_data_path, supply_demand_path)
            
            # Save as HTML
            output_path = os.path.join(output_dir, f"{chart_name}.html")
            fig.write_html(output_path)
            
            print(f"✓ Saved to {output_path}")
        except Exception as e:
            print(f"✗ Error: {e}")
    
    print()
    print(f"✓ All charts generated in {output_dir}/")


def list_available_charts() -> List[str]:
    """List all available chart names"""
    return sorted(CHART_REGISTRY.keys())


def get_chart_info(chart_name: str) -> Dict[str, str]:
    """Get information about a specific chart"""
    if chart_name not in CHART_REGISTRY:
        raise ValueError(f"Unknown chart: {chart_name}")
    
    processor_class, method_name = CHART_REGISTRY[chart_name]
    
    # Get method info without instantiating processor to avoid file errors
    try:
        method = getattr(processor_class, method_name)
        docstring = method.__doc__ or 'No description available'
    except:
        docstring = 'Chart description available'
    
    return {
        'name': chart_name,
        'processor': processor_class.__name__,
        'method': method_name,
        'docstring': docstring
    }


# CLI interface
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python chart_data_processors.py list")
        print("  python chart_data_processors.py generate <chart_name>")
        print("  python chart_data_processors.py generate-all")
        print("  python chart_data_processors.py info <chart_name>")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "list":
        print("Available charts:")
        for name in list_available_charts():
            print(f"  - {name}")
    
    elif command == "generate" and len(sys.argv) >= 3:
        chart_name = sys.argv[2]
        fig = generate_chart(chart_name)
        output_path = f"{chart_name}.html"
        fig.write_html(output_path)
        print(f"✓ Chart saved to {output_path}")
    
    elif command == "generate-all":
        generate_all_charts()
    
    elif command == "info" and len(sys.argv) >= 3:
        chart_name = sys.argv[2]
        info = get_chart_info(chart_name)
        print(f"Chart: {info['name']}")
        print(f"Processor: {info['processor']}")
        print(f"Method: {info['method']}")
        print(f"Description: {info['docstring']}")
    
    else:
        print("Unknown command")
        sys.exit(1)

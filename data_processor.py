import pandas as pd
from typing import Union
import io

class DataProcessor:
    """Handle data loading and processing"""
    
    def load_data(self, file) -> pd.DataFrame:
        """Load data from uploaded file"""
        
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file)
        else:
            raise ValueError(f"Unsupported file format: {file.name}")
        
        # Try to parse date columns
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col])
                except:
                    pass
        
        return df
    
    def prepare_for_chart(self, df: pd.DataFrame, config: dict) -> pd.DataFrame:
        """Prepare data based on chart configuration"""
        
        # Filter columns if specified
        columns_to_keep = [config.get('x_column')]
        columns_to_keep.extend(config.get('y_columns', []))
        
        # Remove None values
        columns_to_keep = [col for col in columns_to_keep if col is not None]
        
        if columns_to_keep:
            df = df[columns_to_keep].copy()
        
        # Remove NaN values
        df = df.dropna()
        
        return df

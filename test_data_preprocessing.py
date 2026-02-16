"""
Test script for AI chart generation with long-format data
"""
import pandas as pd
import json

# Simulate the data structure from the CSV
test_data = {
    'date': ['2022-01-01', '2022-02-01', '2022-01-01', '2022-02-01'],
    'variable': ['Other Services', 'Other Services', 'Transportation Services', 'Transportation Services'],
    'value': [0.0387, 0.1764, -0.0879, 0.0682]
}

df = pd.DataFrame(test_data)

print("Original data (long format):")
print(df.head())
print(f"\nColumns: {df.columns.tolist()}")

# Test pivot
if 'variable' in df.columns and 'value' in df.columns and 'date' in df.columns:
    print("\n✅ Detected long-format data")
    df_wide = df.pivot(index='date', columns='variable', values='value').reset_index()
    df_wide.columns.name = None
    
    print("\nPivoted data (wide format):")
    print(df_wide.head())
    print(f"\nNew columns: {df_wide.columns.tolist()}")
    
    # Test chart config
    config = {
        'chart_type': 'line',
        'x_column': 'date',
        'y_columns': ['Other Services', 'Transportation Services'],
        'title': 'Services Over Time',
        'x_label': 'Date',
        'y_label': 'Value',
        'show_legend': True
    }
    
    print("\nChart config:")
    print(json.dumps(config, indent=2))
    
    # Validate
    assert config['x_column'] in df_wide.columns, "X column not in data"
    for col in config['y_columns']:
        assert col in df_wide.columns, f"Y column {col} not in data"
    
    print("\n✅ All validations passed!")
else:
    print("\n❌ Not long-format data")

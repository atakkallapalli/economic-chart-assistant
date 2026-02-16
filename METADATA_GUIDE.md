# Metadata/Stylesheet & AI Interactive Charting Guide

## Overview
Enhanced charting capabilities with metadata/stylesheet support and AI-powered interactive chart building.

## Features

### 1. Metadata/Stylesheet Upload
Upload JSON or YAML files to define chart styling and configuration.

### 2. Reference Image Upload (AI Training)
Upload an example chart image to train the AI on your desired visual style and format.

### 3. AI Interactive Chart Builder
Build charts using natural language prompts combined with data, metadata, and reference images.

#### Metadata Structure

```json
{
  "chart": {
    "type": "line|bar|scatter|area",
    "x_column": "column_name",
    "y_columns": ["col1", "col2"]
  },
  "style": {
    "title": "Chart Title",
    "x_label": "X Axis Label",
    "y_label": "Y Axis Label",
    "colors": ["#color1", "#color2"],
    "palette": "business|economic|federal|minimal",
    "show_legend": true
  },
  "annotations": [
    {
      "x": "value",
      "y": value,
      "text": "Annotation Text",
      "showarrow": true
    }
  ],
  "custom": {
    "data_source": "Source Name",
    "notes": "Additional notes"
  }
}
```

### 2. AI Interactive Chart Builder
Build charts using natural language prompts combined with data and metadata.

#### How It Works
1. Upload your data (CSV/Excel)
2. Optionally upload metadata/stylesheet (JSON/YAML)
3. Describe the chart you want in natural language
4. AI analyzes prompt, data, and metadata to build the chart

#### Example Prompts
```
"Show GDP growth and inflation trends over time with blue and red colors"
"Create a bar chart comparing quarterly revenue across regions"
"Display correlation heatmap for all economic indicators"
"Line chart of unemployment rate with annotations for policy changes"
```

## Usage Guide

### Step 1: Upload Data
Navigate to **📁 Upload Data & Create Chart** and upload your CSV or Excel file.

### Step 2: Upload Metadata (Optional)
Upload a JSON or YAML metadata file with your chart styling preferences.

### Step 3: Upload Reference Image (Optional)
Upload an example chart image to train the AI on your desired visual style.

### Step 4: AI Chart Builder
Use the **🤖 AI Chart Builder** section:
1. Describe your desired chart in the text area
2. Click **✨ Build Chart with AI**
3. AI will analyze prompt, data, metadata, and reference image to generate the chart

### Step 5: Manual Chart Creation (Alternative)
If you prefer manual control:
1. Select X-axis and Y-axis columns
2. Choose chart type
3. Set date range
4. Add custom title
5. Click **🚀 Generate Chart**

## Metadata Examples

### Example 1: Economic Dashboard
```json
{
  "chart": {
    "type": "line",
    "x_column": "date",
    "y_columns": ["gdp_growth", "inflation"]
  },
  "style": {
    "title": "Economic Indicators Dashboard",
    "x_label": "Time Period",
    "y_label": "Percentage (%)",
    "colors": ["#2E86AB", "#A23B72"],
    "palette": "economic",
    "show_legend": true
  }
}
```

### Example 2: Simple Bar Chart
```yaml
chart:
  type: bar
  x_column: region
  y_columns:
    - revenue
    - profit

style:
  title: "Regional Performance"
  colors:
    - "#1f77b4"
    - "#ff7f0e"
  show_legend: true
```

### Example 3: With Annotations
```json
{
  "chart": {
    "type": "line",
    "x_column": "date",
    "y_columns": ["stock_price"]
  },
  "style": {
    "title": "Stock Price Analysis",
    "colors": ["#2ca02c"]
  },
  "annotations": [
    {
      "x": "2020-03-15",
      "y": 150,
      "text": "Market Crash",
      "showarrow": true
    },
    {
      "x": "2021-01-01",
      "y": 200,
      "text": "Recovery",
      "showarrow": true
    }
  ]
}
```

## AI Prompt Best Practices

### Good Prompts
✅ "Create a line chart showing GDP and inflation from 2020 to 2024 with blue and red colors"
✅ "Bar chart comparing revenue across all regions, use business color palette"
✅ "Scatter plot of price vs demand with economic styling"

### Avoid
❌ "Make a chart" (too vague)
❌ "Show everything" (not specific)
❌ "Chart with data" (no details)

## Integration with Existing Features

### Works With
- Date range filtering
- Chart customization
- AI analysis and summaries
- Persona-based summaries
- Q&A functionality
- Chart saving and reports

### Metadata Priority
When both metadata and manual selections are provided:
1. Metadata provides base styling
2. Manual selections override metadata
3. AI prompts can override both

## Troubleshooting

### Metadata Not Loading
- Check JSON/YAML syntax
- Ensure file extension is correct (.json, .yaml, .yml)
- Verify all required fields are present

### AI Chart Builder Fails
- Ensure data is uploaded first
- Check LLM provider is configured
- Try simpler prompts
- Fall back to manual chart creation

### Colors Not Applying
- Use hex color codes (#RRGGBB)
- Ensure colors array matches number of y_columns
- Check metadata structure

## API Reference

### MetadataHandler
```python
# Parse metadata
metadata = metadata_handler.parse_metadata(file_content, file_type)

# Merge with config
merged_config = metadata_handler.merge_with_config(base_config, metadata)
```

### LLMHandler
```python
# Build chart interactively
config = llm_handler.build_chart_interactively(
    user_prompt="Show GDP trends",
    data=dataframe,
    provider="AWS Bedrock",
    metadata=metadata_dict
)
```

## Sample Files
- `sample_metadata.json` - JSON metadata example
- `sample_metadata.yaml` - YAML metadata example

## Benefits

1. **Consistency**: Reuse styling across multiple charts
2. **Efficiency**: Define once, apply to many charts
3. **Flexibility**: Override with AI prompts or manual selections
4. **Documentation**: Metadata serves as chart documentation
5. **Collaboration**: Share metadata files across team
6. **Visual Training**: Reference images help AI match your exact style
7. **Accuracy**: AI learns from examples to produce better results

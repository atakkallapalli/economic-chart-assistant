# Economic Chart Assistant - Complete Guide

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Initialize database
python init_db.py

# Run the enhanced version (recommended)
streamlit run app_enhanced.py

# Or run the original version
streamlit run app_final.py
```

## 📊 Overview

Two versions available:
- **app_enhanced.py**: FRED-style professional charts with advanced customization
- **app_final.py**: Streamlined AI-powered chart creation

Both share the same database and can be used interchangeably.

## ✨ Key Features

### app_enhanced.py (Professional)
- **4-Tab Interface**: Data, Appearance, Axes, Advanced
- **FRED-Style Controls**: Line width, styles, markers, colors
- **Chart Dimensions**: Custom width/height (600-1600 × 400-1200 px)
- **Legend Positioning**: 8 positions (top, bottom, left, right, corners)
- **Axis Configuration**: Linear/log scale, manual ranges, grid controls
- **Data Transformations**: % Change, Difference, YoY, Annual Rate
- **Annotations**: Recession shading, reference lines
- **Export Options**: PNG, SVG, PDF, CSV
- **Saved Charts**: View and analyze previously saved charts
- **AI Analysis**: Quick analysis, persona summaries, Q&A

### app_final.py (Streamlined)
- **Quick Upload**: Fast data upload and chart creation
- **AI Chart Builder**: Natural language chart generation
- **Metadata Support**: JSON/YAML styling configuration
- **Auto-Detection**: Long/wide format data handling
- **AI Customization**: Natural language chart modifications

## 📖 Usage Guide

### Creating Charts (app_enhanced.py)

#### 1. Upload Data
```
1. Click "📁 Upload Data" in sidebar
2. Upload CSV/Excel file
3. (Optional) Upload metadata JSON/YAML
```

#### 2. AI Chart Builder
```
Describe chart: "Show GDP and inflation trends with blue and red colors"
Click "✨ Build Chart with AI"
```

#### 3. Manual Chart Creation

**Data Tab:**
- Select X-axis (date) and Y-axis (numeric) columns
- Choose chart type: line, bar, scatter, area, stacked
- Use quick date presets: 1Y, 5Y, 10Y, Max

**Appearance Tab:**
- Line Width: 1-5 px
- Line Style: solid, dash, dot, dashdot
- Markers: Toggle on/off, size 4-12 px
- Colors: Individual picker per series
- Dimensions: Custom width × height
- Legend: 8 positions

**Axes Tab:**
- Custom X/Y labels
- Y-Scale: Linear or Logarithmic
- Grid controls (X and Y separate)
- Manual Y-axis range

**Advanced Tab:**
- Transform: None, % Change, Difference, YoY, Annual Rate
- Aggregation: None, Average, Sum, End of Period
- Recession Shading: COVID-19, Great Recession
- Reference Lines: Custom value and label

#### 4. Save Chart
```
Expand "💾 Save Chart"
Select category: Inflation, Growth, Employment, etc.
Click "💾 Save"
```

### Viewing Saved Charts

```
1. Click "💾 Saved Charts" in sidebar
2. Select chart from dropdown
3. View chart and use AI analysis tools
```

### AI Analysis Features

**Quick Analysis** (100 words):
- Main trend
- Key insight
- Implication

**Persona Summaries** (80 words each):
- Executive: Business perspective
- Economist: Technical analysis
- General Public: Simplified explanation

**Q&A** (60 words):
- Ask specific questions about the chart
- Get concise answers

## 🎨 FRED-Style Features

Inspired by https://fred.stlouisfed.org/

| Feature | FRED | app_enhanced.py |
|---------|------|-----------------|
| Line width/style | ✅ | ✅ |
| Custom colors | ✅ | ✅ |
| Chart dimensions | ✅ | ✅ |
| Legend positioning | ✅ 4 | ✅ 8 |
| Log scale | ✅ | ✅ |
| Custom labels | ✅ | ✅ |
| Grid controls | ✅ | ✅ |
| Manual ranges | ✅ | ✅ |
| Transformations | ✅ | ✅ 5 types |
| Recession shading | ✅ | ✅ |
| Reference lines | ✅ | ✅ |
| Date presets | ✅ | ✅ |
| Export PNG/SVG/PDF | ✅ | ✅ |
| **AI Features** | ❌ | ✅ |

## 🔧 Configuration

### Environment Variables (.env)
```env
# AWS Bedrock (recommended)
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret

# Alternative: Anthropic
ANTHROPIC_API_KEY=your_key

# Alternative: OpenAI
OPENAI_API_KEY=your_key
```

### Metadata/Stylesheet (Optional)

Upload JSON/YAML files to provide chart styling:

```json
{
  "chart_identity": {
    "title": "PCE Inflation",
    "chart_type": "line",
    "description": "Core and headline PCE"
  },
  "visual_styling": {
    "colors": ["#1f77b4", "#d62728"],
    "line_width": 3
  }
}
```

## 💡 Pro Tips

### For Presentations
- Dimensions: 1400×800
- Line Width: 3-4
- High-contrast colors
- Export: PNG or PDF

### For Reports
- Dimensions: 1000×600
- Add subtitle with source
- Professional color palette
- Export: SVG (scalable)

### For Analysis
- Use log scale for exponential data
- Apply transformations (% Change, YoY)
- Add recession shading for context
- Export: CSV for further analysis

### Color Recommendations
- GDP Growth: #2ca02c (green)
- Inflation: #d62728 (red)
- Unemployment: #ff7f0e (orange)
- Interest Rates: #1f77b4 (blue)

## 🔍 Troubleshooting

### Charts not displaying
- Check data format (wide vs long)
- Verify date column is recognized
- Try different chart type

### Export not working
```bash
pip install kaleido
```

### Performance issues
- Reduce chart dimensions
- Limit date range for large datasets
- Disable markers for dense data

### Database errors
```bash
python init_db.py
```

## 📊 Data Format

### Wide Format (Preferred)
```csv
date,gdp,inflation
2020-01-01,2.5,3.0
2020-02-01,2.7,3.2
```

### Long Format (Auto-converted)
```csv
date,variable,value
2020-01-01,gdp,2.5
2020-01-01,inflation,3.0
2020-02-01,gdp,2.7
```

## 🎯 Use Cases

### Use app_enhanced.py for:
- Professional presentations
- Publication-ready charts
- Detailed customization
- Print media
- FRED-style charts

### Use app_final.py for:
- Quick analysis
- AI-first workflows
- Simple visualizations
- Rapid prototyping

## 🆘 Support

### Common Issues

**"No module named 'streamlit'"**
```bash
pip install -r requirements.txt
```

**"AWS credentials not found"**
- Check .env file exists
- Verify credentials are correct

**Charts not generating**
- Verify data uploaded successfully
- Check columns selected properly
- Try sample data first

## 📈 Examples

### Example 1: CPI Inflation Chart
```
1. Upload CPALTT01USM657N.csv
2. Data Tab: Select observation_date (X), CPALTT01USM657N (Y)
3. Click "10Y" preset
4. Appearance Tab: Line Width 3, Color #1f77b4
5. Advanced Tab: Transform "% Change from Year Ago", Recession Shading On
6. Generate Chart
7. Export PNG
```

### Example 2: Multi-Series Comparison
```
1. Upload data with GDP, Inflation, Unemployment columns
2. Data Tab: Select all Y columns, Chart Type "line"
3. Appearance Tab: Assign colors (green, red, orange)
4. Axes Tab: Y-Scale Linear, Show grids
5. Advanced Tab: Add reference line at 2.0 (Fed target)
6. Generate Chart
```

### Example 3: AI Chart Builder
```
1. Upload data
2. AI Chart Builder: "Show GDP growth and inflation trends over time with blue and red colors"
3. Click "✨ Build Chart with AI"
4. AI generates chart automatically
5. Save chart
```

## 🔄 Version Comparison

| Feature | app_final.py | app_enhanced.py |
|---------|:------------:|:---------------:|
| Quick Charts | ✅ | ✅ |
| AI Features | ✅ | ✅ |
| Line Styling | ❌ | ✅ |
| Chart Dimensions | ❌ | ✅ |
| Legend Positions | 1 | 8 |
| Log Scale | ❌ | ✅ |
| Transformations | ❌ | ✅ |
| Annotations | ❌ | ✅ |
| Export Formats | ❌ | ✅ 4 |
| Date Presets | ❌ | ✅ |
| Saved Charts | ❌ | ✅ |

## 📝 Changelog

### Enhanced Version (app_enhanced.py)
- Added FRED-style customization (32 features)
- Implemented 4-tab interface
- Added saved charts functionality
- Optimized UI for less scrolling
- Reduced AI summary length (60-100 words)
- Integrated save chart into chart creation
- Minimized sidebar navigation

### Original Version (app_final.py)
- AI Chart Builder
- Metadata/Stylesheet support
- Long-format data handling
- Series identifier selection
- Stacked chart types
- Component color selection

## 🎓 Learning Resources

- **FRED**: https://fred.stlouisfed.org/
- **Plotly**: https://plotly.com/python/
- **Streamlit**: https://docs.streamlit.io/

## 📄 License

MIT License

---

**Quick Reference:**
- Run: `streamlit run app_enhanced.py`
- Upload: CSV/Excel + optional JSON/YAML metadata
- Create: Use tabs (Data → Appearance → Axes → Advanced)
- Save: Expand "💾 Save Chart" after generation
- View: Click "💾 Saved Charts" in sidebar
- Analyze: Use AI buttons (Quick Analysis, Summaries, Q&A)

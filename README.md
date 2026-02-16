# Economic Chart Assistant 📊

A comprehensive AI-powered web application for economic researchers to create, analyze, and summarize charts using natural language processing.

## 🚀 Features

### 📊 **Data Analysis**
- **Upload Data & Create Chart**: Upload CSV/Excel files and generate charts with column selection
- **Metadata/Stylesheet Support**: Upload JSON/YAML files to define chart styling and configuration
- **AI Interactive Chart Builder**: Build charts using natural language prompts combined with data and metadata
- **Upload Image & Data**: Analyze existing chart images with corresponding datasets
- **AI-Powered Analysis**: Get insights, summaries, and answers about your charts

### 🔍 **Pre-Built Economic Charts**
- **PCE Inflation Analysis**: Core and headline PCE inflation charts
- **Component Breakdown**: Detailed PCE component analysis
- **Timing Analysis**: Economic timing and trend analysis
- **Supply/Demand Charts**: Economic supply and demand visualizations

### 🤖 **AI-Powered Features**
- **Quick Analysis**: Get instant insights from any chart
- **Persona Summaries**: Executive, Economist, and General Public perspectives
- **Natural Language Q&A**: Ask questions about your charts
- **Chart Customization**: Modify charts using natural language commands

### 💾 **Data Management**
- **Automatic Saving**: All charts and customizations saved to database
- **Time Series Processing**: Auto-converts data to proper time series format
- **Date Range Filtering**: Select specific time periods for analysis
- **Smart Column Detection**: Automatically identifies date and numeric columns

## 🛠️ Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Quick Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd charting_assistant
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure environment variables**
```bash
cp .env.example .env
```

Edit `.env` file with your API keys:
```env
# AWS Bedrock (recommended)
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret

# Alternative: Anthropic API
ANTHROPIC_API_KEY=your_anthropic_key

# Alternative: OpenAI API
OPENAI_API_KEY=your_openai_key
```

4. **Initialize database**
```bash
python init_db.py
```

5. **Run the application**
```bash
streamlit run app_final.py
```

6. **Open your browser**
Navigate to `http://localhost:8501`

## 📖 Quick Start Guide

### 1. Upload Data & Create Chart
1. Click **"📁 Upload Data & Create Chart"** in the left sidebar
2. Upload your CSV/Excel file or use sample data
3. **(Optional)** Upload metadata/stylesheet (JSON/YAML) for styling
4. **Option A - AI Chart Builder**:
   - Describe your chart in natural language
   - Click **"✨ Build Chart with AI"**
   - AI analyzes prompt, data, and metadata to generate chart
5. **Option B - Manual Creation**:
   - Select X-axis (date/time) and Y-axis (numeric) columns
   - Set date range if needed
   - Add custom chart title (optional)
   - Click **"🚀 Generate Chart"**
6. Use AI tools in the right panel for analysis

### 2. Analyze Existing Charts
1. Click **"🖼️ Upload Image & Data"** in the left sidebar
2. Upload a chart image (PNG/JPG)
3. Upload the corresponding data file (CSV/Excel)
4. Use AI analysis tools for insights and summaries

### 3. Use Pre-Built Charts
1. Select category from **"🔍 Filters"** (PCE Inflation, Components, etc.)
2. Choose specific chart and chart type
3. Set date range
4. Click **"🚀 Generate Chart"**
5. Customize using AI tools in the right panel

## 🔧 Configuration

### LLM Providers
The application supports three LLM providers:

#### AWS Bedrock (Recommended)
- Most reliable and cost-effective
- Uses Claude 3 Sonnet model
- Requires AWS account and Bedrock access

#### Anthropic API
- Direct API access to Claude
- Good for development and testing
- Requires Anthropic API key

#### OpenAI API
- Uses GPT-4 model
- Alternative option
- Requires OpenAI API key

### Environment Variables
```env
# AWS Configuration
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key

# API Keys
ANTHROPIC_API_KEY=your_anthropic_key
OPENAI_API_KEY=your_openai_key
```

## 🔍 Troubleshooting

### Common Issues

#### "No module named 'streamlit'"
```bash
pip install -r requirements.txt
```

#### "AWS credentials not found"
- Ensure `.env` file exists with AWS credentials
- Or configure AWS CLI: `aws configure`

#### "API key not found"
- Check `.env` file has the correct API key
- Ensure no extra spaces or quotes around keys

#### Charts not generating
- Verify data has been uploaded successfully
- Check that columns are properly selected
- Try sample data first
- Check LLM provider selection

#### Database errors
```bash
# Reinitialize database
python init_db.py
```

### Performance Tips
- Use AWS Bedrock for best performance and cost
- Keep data files under 10MB for optimal processing
- Use specific date ranges for faster chart generation
- Clear browser cache if interface issues occur

## 📊 Usage Examples

### Chart Creation Examples
```
"PCE Inflation Trends 2020-2024"
"GDP Growth Analysis"
"Consumer Spending by Category"
"Employment vs Inflation Correlation"
```

### AI Chart Builder Examples
```
"Show GDP growth and inflation trends over time with blue and red colors"
"Create a bar chart comparing quarterly revenue across regions"
"Display correlation heatmap for all economic indicators"
"Line chart of unemployment rate with annotations for policy changes"
```

### Customization Examples
```
"Change the title to 'Recent Economic Trends'"
"Use blue and red colors for the lines"
"Add grid lines to make it easier to read"
"Make the chart title larger and bold"
```

### Analysis Questions
```
"What are the main trends in this inflation data?"
"How do these indicators correlate with economic policy?"
"What policy implications can we draw from these charts?"
"Compare the performance before and after 2020"
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Review environment variable configuration
3. Verify API keys are correctly set
4. Test with sample data first

## 🔄 Updates

The application automatically saves all charts and configurations. Recent updates include:
- **Metadata/Stylesheet Support**: Upload JSON/YAML files for chart styling
- **AI Interactive Chart Builder**: Build charts using natural language prompts
- Enhanced time series data processing
- Smart column detection and filtering
- Improved AI analysis reliability
- Better chart customization options
- Streamlined user interface

For detailed metadata documentation, see [METADATA_GUIDE.md](METADATA_GUIDE.md)
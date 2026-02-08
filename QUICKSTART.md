# Quick Start Guide 🚀

Get the Economic Chart Assistant running in 5 minutes!

## ⚡ Instant Setup

### 1. Install Dependencies
```bash
pip install streamlit plotly pandas numpy boto3 anthropic openai python-dotenv sqlite3
```

### 2. Configure API Keys
Create `.env` file:
```env
# Choose ONE provider:

# Option 1: AWS Bedrock (Recommended)
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret

# Option 2: Anthropic
ANTHROPIC_API_KEY=your_anthropic_key

# Option 3: OpenAI
OPENAI_API_KEY=your_openai_key
```

### 3. Initialize & Run
```bash
python init_db.py
streamlit run app_final.py
```

### 4. Open Browser
Navigate to: `http://localhost:8501`

## 🎯 First Steps

### Upload & Analyze Data
1. Click **"📁 Upload Data & Create Chart"**
2. Upload CSV file or click **"📊 Use Sample Data"**
3. **Select Columns**:
   - X-axis: Choose date/time column
   - Y-axis: Select numeric columns to plot
4. **Set Date Range**: Choose start and end dates
5. **Add Title**: Optional custom chart title
6. Click **"🚀 Generate Chart"**
7. Use AI tools in right panel

### Try Pre-Built Charts
1. Select **"PCE Inflation"** from filters
2. Choose any chart from dropdown
3. Click **"🚀 Generate Chart"**
4. Ask questions or get summaries

## 🔧 Quick Fixes

### No Charts Appearing?
- Check your API keys in `.env`
- Verify LLM provider selection
- Try sample data first

### Database Issues?
```bash
python init_db.py
```

### Import Errors?
```bash
pip install -r requirements.txt
```

## 📊 New Features

**Smart Column Selection:**
- X-axis shows only date/time columns
- Y-axis shows only numeric columns
- Date range selectors with min/max validation

**Time Series Processing:**
- Auto-converts timestamps to proper format
- Handles Excel date formats
- Smart data type detection

**Enhanced Charts:**
- Custom titles
- Legend positioning at bottom
- Metrics displayed on Y-axis
- Date range filtering

## 🎉 You're Ready!

The app now features intelligent column detection, time series processing, and reliable AI analysis. Upload your economic data and start exploring!
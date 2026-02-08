# Deployment Guide 🚀

Complete guide for deploying Economic Chart Assistant to production environments with latest features.

## 🌐 Deployment Options

### 1. Streamlit Cloud (Easiest)
**Best for:** Quick deployment, small teams, prototypes

#### Steps:
1. **Push to GitHub**
```bash
git add .
git commit -m "Deploy Economic Chart Assistant v2.0"
git push origin main
```

2. **Deploy on Streamlit Cloud**
- Go to [share.streamlit.io](https://share.streamlit.io)
- Connect GitHub repository
- Select `app_final.py` as main file
- Add environment variables in "Advanced settings"

3. **Configure Environment Variables**
```
AWS_REGION = us-east-1
AWS_ACCESS_KEY_ID = your_aws_key
AWS_SECRET_ACCESS_KEY = your_aws_secret
```

4. **Deploy**
- Click "Deploy"
- App will be available at `https://your-app.streamlit.app`

---

### 2. Docker Deployment
**Best for:** Containerized environments, consistent deployments

#### Create Dockerfile:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Initialize database
RUN python init_db.py

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run application
CMD ["streamlit", "run", "app_final.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

#### Build and Run:
```bash
# Build image
docker build -t economic-chart-assistant .

# Run container
docker run -p 8501:8501 \
  -e AWS_REGION=us-east-1 \
  -e AWS_ACCESS_KEY_ID=your_key \
  -e AWS_SECRET_ACCESS_KEY=your_secret \
  economic-chart-assistant
```

---

### 3. AWS EC2 Deployment
**Best for:** Full control, custom configurations, enterprise

#### Launch EC2 Instance:
```bash
# Choose Ubuntu 20.04 LTS
# Instance type: t3.medium or larger (for data processing)
# Security group: Allow HTTP (80), HTTPS (443), SSH (22), Custom (8501)
```

#### Server Setup:
```bash
# Connect to instance
ssh -i your-key.pem ubuntu@your-ec2-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and dependencies
sudo apt install python3-pip python3-venv nginx -y

# Clone repository
git clone your-repo-url
cd charting_assistant

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Initialize database
python init_db.py
```

#### Configure Environment:
```bash
# Create environment file
sudo nano /etc/environment

# Add variables:
AWS_REGION="us-east-1"
AWS_ACCESS_KEY_ID="your_key"
AWS_SECRET_ACCESS_KEY="your_secret"
```

#### Create Systemd Service:
```bash
sudo nano /etc/systemd/system/chart-assistant.service
```

```ini
[Unit]
Description=Economic Chart Assistant v2.0
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/charting_assistant
Environment=PATH=/home/ubuntu/charting_assistant/venv/bin
ExecStart=/home/ubuntu/charting_assistant/venv/bin/streamlit run app_final.py --server.port=8501 --server.address=0.0.0.0
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

#### Start Service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable chart-assistant
sudo systemctl start chart-assistant
sudo systemctl status chart-assistant
```

---

## 🔒 Security Configuration

### Environment Variables
Never commit API keys to version control:

```bash
# Use environment variables
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret

# Or use AWS IAM roles (recommended for EC2)
```

### AWS IAM Role (Recommended)
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "bedrock:InvokeModel"
            ],
            "Resource": "arn:aws:bedrock:*:*:foundation-model/anthropic.claude-3-sonnet-20240229-v1:0"
        }
    ]
}
```

---

## 📊 Performance Optimization

### Resource Requirements
- **Minimum**: 2 CPU, 4GB RAM, 10GB storage
- **Recommended**: 4 CPU, 8GB RAM, 20GB storage
- **Heavy Usage**: 8 CPU, 16GB RAM, 50GB storage

### Application Optimization
```python
# In app_final.py - already implemented
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_data(file_path):
    return pd.read_csv(file_path)

@st.cache_resource
def get_llm_handler():
    return LLMHandler()
```

### Database Optimization
```bash
# Regular database maintenance
sqlite3 charts.db "VACUUM;"
sqlite3 charts.db "REINDEX;"
```

---

## 🔄 Backup & Recovery

### Database Backup
```bash
#!/bin/bash
# backup.sh
DATE=$(date +%Y%m%d_%H%M%S)
cp charts.db "backups/charts_$DATE.db"
find backups/ -name "charts_*.db" -mtime +7 -delete
```

### Automated Backups
```bash
# Add to crontab
0 2 * * * /path/to/backup.sh
```

---

## 📊 Monitoring & Logging

### Application Logs
```bash
# View logs
sudo journalctl -u chart-assistant -f

# Log rotation
sudo nano /etc/logrotate.d/chart-assistant
```

### Health Checks
```bash
#!/bin/bash
# monitor.sh
while true; do
    if ! curl -f http://localhost:8501/_stcore/health; then
        sudo systemctl restart chart-assistant
        echo "$(date): Restarted chart-assistant" >> /var/log/monitor.log
    fi
    sleep 60
done
```

---

## 🎆 New Features Deployment Notes

### Enhanced Data Processing
- **Time Series Auto-Conversion**: Handles Excel timestamps automatically
- **Smart Column Detection**: No manual column type specification needed
- **Date Range Filtering**: Built-in date validation and filtering

### Improved User Interface
- **Column Selection**: Visual dropdowns for X/Y axis selection
- **Custom Titles**: User-defined chart titles
- **Legend Positioning**: Professional bottom-center legends

### Reliable AI Features
- **Direct LLM Calls**: Simplified error-free AI integration
- **Fallback Responses**: Graceful handling of AI service failures
- **Multi-Provider Support**: AWS Bedrock, Anthropic, OpenAI

---

## ✅ Deployment Checklist

- [ ] Environment variables configured
- [ ] Database initialized (`python init_db.py`)
- [ ] SSL certificate installed (production)
- [ ] Firewall configured (port 8501 open)
- [ ] Monitoring setup
- [ ] Backup system configured
- [ ] Health checks implemented
- [ ] Load testing completed
- [ ] Documentation updated
- [ ] Team access configured
- [ ] AI provider API keys tested
- [ ] Sample data upload tested
- [ ] Chart generation verified
- [ ] AI analysis features tested

---

## 🎆 Version 2.0 Features

### 📊 **Data Processing**
- Auto time series conversion
- Smart column type detection
- Excel/CSV format support
- Date range validation

### 🎨 **User Interface**
- Visual column selection
- Custom chart titles
- Professional legend placement
- Metric-focused Y-axis labels

### 🤖 **AI Integration**
- Reliable persona summaries
- Error-free quick analysis
- Natural language Q&A
- Chart customization

### 💾 **Data Management**
- Enhanced database schema
- Filtered data persistence
- Chart configuration storage
- User preference tracking

Your Economic Chart Assistant v2.0 is now ready for production! 🎉
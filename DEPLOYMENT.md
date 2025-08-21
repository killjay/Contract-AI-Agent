# Streamlit Cloud Deployment Guide

## 🚀 Quick Deployment Steps

### 1. Create New App in Streamlit Cloud
- Go to [share.streamlit.io](https://share.streamlit.io)
- Sign in with GitHub
- Click "New app"

### 2. Repository Settings
```
Repository: killjay/Contract-AI-Agent
Branch: main
Main file path: standalone_ui.py
App URL: contract-ai-agent (or your choice)
```

### 3. Configure Secrets (CRITICAL!)
In **Advanced settings** → **Secrets**, paste:

```toml
ANTHROPIC_API_KEY = "your_actual_claude_api_key_here"
LLM_MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = "20000"
TESSERACT_CMD = "/usr/bin/tesseract"
```

**⚠️ IMPORTANT**: Replace `your_actual_claude_api_key_here` with your real Claude API key!

### 4. Deploy
Click "Deploy!" and wait 2-5 minutes.

## 🔧 Troubleshooting

### "Claude API key is required" Error
✅ **Solution**: Configure secrets properly in Streamlit Cloud
1. Go to your app settings (gear icon)
2. Click "Secrets" tab
3. Add your ANTHROPIC_API_KEY
4. Save and restart app

### App Won't Start
✅ **Check**:
- API key is correctly set in secrets
- No extra quotes in the API key
- All required packages are in requirements.txt

### OCR Not Working
✅ **Solution**: The packages.txt file should install Tesseract automatically

## 📱 Expected Result
Your app will be available at: `https://your-app-name.streamlit.app`

## 💡 Tips
- Free tier has 1GB RAM limit
- App may sleep after inactivity (30-60 seconds to wake up)
- Monitor Claude API usage for costs
- Keep document uploads under 200MB for best performance

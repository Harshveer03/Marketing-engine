# 🚀 Personal Marketing Engine

A complete AI-powered marketing content generation system that creates personalized blogs, social media posts, and performance analytics based on your business documents.

## ✨ Features

- **📄 Document-Based Setup**: Upload your business PDF and the system automatically configures itself
- **🤖 AI Content Generation**: Creates blogs, LinkedIn posts, Twitter content, and YouTube scripts
- **📊 Performance Analytics**: Tracks and analyzes content performance with AI insights
- **🌐 Web Dashboard**: Beautiful, responsive web interface for easy management
- **📱 Multi-Platform**: Generates content optimized for different social media platforms
- **📈 Trend Integration**: Fetches and incorporates latest industry trends
- **💾 Export Functionality**: Export all your content in various formats

## 🏗️ Architecture

```
Marketing Engine
├── 📄 Document Processing (PDF → Knowledge Base)
├── 🎯 Niche Extraction (AI-powered ICP analysis)
├── 📰 Trend Fetching (Real-time industry insights)
├── ✍️ Content Generation (Blogs, Social Media)
├── 📊 Performance Analytics (AI-driven insights)
└── 🔄 Feedback Loop (Continuous improvement)
```

## 🚀 Quick Start

### Option 1: Automated Installation

**Windows:**

```bash
# Run the installer
install.bat

# Start the web dashboard
python app.py
```

**Mac/Linux:**

```bash
# Make installer executable and run
chmod +x install.sh
./install.sh

# Start the web dashboard
python app.py
```

### Option 2: Manual Installation

1. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Create Environment File**

   ```bash
   # Create .env file with your API keys
   GOOGLE_API_KEY=your_google_api_key
   OLLAMA_BASE_URL=http://localhost:11434
   SERPAPI_KEY=your_serpapi_key
   ```

3. **Setup Your Business**

   ```bash
   # Place your business document in /data folder
   # Then run setup
   python setup.py
   ```

4. **Start the Dashboard**
   ```bash
   python app.py
   # Open http://localhost:5000
   ```

## 📋 Requirements

### System Requirements

- Python 3.8+
- 4GB RAM minimum
- Internet connection for AI APIs

### API Keys Required

- **Google Gemini API**: For AI content generation
- **SerpAPI**: For trend fetching (optional)
- **Ollama**: For local embeddings (optional)

### Supported Document Formats

- PDF files
- DOCX files
- TXT files

## 🎯 How It Works

### 1. **Initial Setup**

- Upload your business document (pitch deck, business plan, etc.)
- System extracts your niche, target audience, and value proposition
- Creates a personalized knowledge base using vector embeddings

### 2. **Content Generation**

- **Blogs**: Long-form strategic content (700-1000 words)
- **LinkedIn**: Professional thought leadership posts
- **Twitter**: Concise, engaging tweets with hashtags
- **YouTube**: Video scripts with SEO-optimized descriptions

### 3. **Performance Tracking**

- Simulated engagement metrics for testing
- AI-powered performance analysis
- Actionable recommendations for improvement

### 4. **Continuous Improvement**

- Feedback loop analyzes what content performs best
- Adapts future content generation based on insights
- Learns from your specific business context

## 🖥️ Web Dashboard

The web dashboard provides:

- **📊 Overview**: Stats and performance metrics
- **⚡ Quick Actions**: One-click content generation
- **📝 Content Management**: View, edit, and organize all content
- **📈 Analytics**: Detailed performance insights
- **💾 Export Tools**: Download content in various formats

### Dashboard Features

- **Responsive Design**: Works on desktop, tablet, and mobile
- **Real-time Updates**: Live content generation status
- **Content Preview**: View full content before publishing
- **Copy to Clipboard**: Easy content sharing
- **Export Options**: JSON, CSV, and formatted exports

## 📁 File Structure

```
marketing_engine/
├── 📄 Core Files
│   ├── app.py                 # Web dashboard
│   ├── setup.py              # Initial configuration
│   ├── marketing_engine.py   # CLI interface
│   └── requirements.txt      # Dependencies
│
├── 🧠 AI Modules
│   ├── blog_generator.py     # Blog content generation
│   ├── post_generator.py     # Social media content
│   ├── trend_fetcher.py      # Industry trend analysis
│   ├── extractor.py          # Business niche extraction
│   └── performance_analyzer.py # AI performance insights
│
├── 📊 Analytics
│   ├── performance_fetcher.py # Metrics collection
│   ├── feedback_loop.py      # Improvement recommendations
│   └── query_helper.py       # Knowledge base queries
│
├── 🎨 Frontend
│   ├── templates/            # HTML templates
│   ├── static/css/          # Stylesheets
│   └── static/js/           # JavaScript
│
├── 📁 Data Directories
│   ├── data/                # Your business documents
│   ├── generated/           # All generated content
│   ├── config/              # System configuration
│   └── vectordb/            # Knowledge base
│
└── 🔧 Setup
    ├── install.bat          # Windows installer
    ├── install.sh           # Mac/Linux installer
    └── README.md            # This file
```

## 🎛️ Configuration

### Environment Variables (.env)

```bash
# Required
GOOGLE_API_KEY=your_google_gemini_api_key

# Optional
OLLAMA_BASE_URL=http://localhost:11434
SERPAPI_KEY=your_serpapi_key_for_trends
```

### System Settings (config/settings.json)

```json
{
  "business_name": "Your Business",
  "setup_date": "2024-01-01T00:00:00",
  "document_processed": "./data/your_document.pdf",
  "target_audience": ["Founders", "Marketers"],
  "value_proposition": "Your unique value proposition"
}
```

## 🔧 Usage Examples

### CLI Usage

```bash
# Generate a blog post
python marketing_engine.py
# Select option 1

# Generate social media content
python marketing_engine.py
# Select option 2

# Run performance analysis
python marketing_engine.py
# Select option 4
```

### Web Dashboard Usage

1. Open http://localhost:5000
2. Click "Generate Blog" for new blog content
3. Click "Generate Social" for social media posts
4. View content in the respective tabs
5. Export content using the export button

### API Usage (Advanced)

```python
from blog_generator import BlogGenerator
from post_generator import ContentPipeline

# Generate blog
generator = BlogGenerator()
blog = generator.run(mode="automatic")

# Generate social content
pipeline = ContentPipeline()
pipeline.run()
```

## 📊 Performance Analytics

The system provides comprehensive analytics:

### Content Metrics

- **Engagement Rates**: Likes, shares, comments
- **Reach Metrics**: Impressions, unique views
- **Conversion Tracking**: Click-through rates
- **Quality Scores**: Time on page, bounce rate

### AI Insights

- **Platform Optimization**: Best performing content types per platform
- **Timing Analysis**: Optimal posting schedules
- **Content Recommendations**: Data-driven suggestions
- **Trend Correlation**: How trends impact performance

### Feedback Loop

- **Performance Patterns**: What content works best
- **Audience Insights**: Who engages most
- **Content Evolution**: How to improve future posts
- **Strategic Recommendations**: High-level marketing advice

## 🔒 Privacy & Security

- **Local Processing**: All data stays on your machine
- **No Data Sharing**: Your business information is never shared
- **Secure APIs**: Encrypted communication with AI services
- **File Security**: Safe document processing and storage

## 🛠️ Troubleshooting

### Common Issues

**Setup fails:**

- Check Python version (3.8+ required)
- Verify API keys in .env file
- Ensure internet connection

**Content generation errors:**

- Verify Google API key is valid
- Check API quotas and limits
- Ensure document is properly formatted

**Dashboard not loading:**

- Check if port 5000 is available
- Try running with `python app.py --port 8000`
- Clear browser cache

### Getting Help

1. Check the error logs in the console
2. Verify all dependencies are installed
3. Ensure API keys are correctly configured
4. Try regenerating with a different document

## 🚀 Advanced Features

### Custom Prompts

Modify the AI prompts in each generator file to customize output style.

### Multiple Businesses

Run separate instances for different businesses by using different directories.

### API Integration

Extend the system to integrate with real social media APIs for automatic posting.

### Custom Analytics

Add your own metrics and KPIs to the performance tracking system.

## 📈 Roadmap

- [ ] Real social media API integration
- [ ] Advanced scheduling features
- [ ] Multi-language support
- [ ] Custom branding options
- [ ] Team collaboration features
- [ ] Advanced analytics dashboard
- [ ] Mobile app companion

## 📄 License

This is a commercial product. See LICENSE.txt for usage terms.

## 🤝 Support

For support and questions:

- Check the troubleshooting section above
- Review the configuration settings
- Ensure all requirements are met

---

**🎉 Congratulations!** You now have a complete, personalized marketing engine that creates content tailored specifically to your business. The system learns from your documents and continuously improves its output based on performance data.

Start generating amazing content today! 🚀

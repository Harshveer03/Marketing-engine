#!/usr/bin/env python3

import os
import sys

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from app import app
    
    print("🚀 Marketing Engine Dashboard Starting...")
    print("📊 Open your browser to: http://localhost:5000")
    print("🔧 Press Ctrl+C to stop the server")
    print("-" * 50)
    
    # Create necessary directories
    os.makedirs('./generated/content/blogs', exist_ok=True)
    os.makedirs('./generated/content/social', exist_ok=True)
    os.makedirs('./generated/analytics', exist_ok=True)
    os.makedirs('./generated/news', exist_ok=True)
    os.makedirs('./generated/topics', exist_ok=True)
    os.makedirs('./config', exist_ok=True)
    
    # Start the Flask app
    app.run(debug=False, host='127.0.0.1', port=5000, use_reloader=False)
    
except Exception as e:
    print(f"❌ Error starting dashboard: {e}")
    print("\n🔧 Troubleshooting:")
    print("1. Make sure all dependencies are installed: pip install -r requirements.txt")
    print("2. Check if port 5000 is available")
    print("3. Verify your .env file has the required API keys")
    input("\nPress Enter to exit...")
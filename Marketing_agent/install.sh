#!/bin/bash

echo "========================================"
echo "   Marketing Engine Installation"
echo "========================================"
echo

echo "Installing Python dependencies..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo
    echo "ERROR: Failed to install dependencies!"
    echo "Please make sure Python and pip are installed."
    exit 1
fi

echo
echo "Creating necessary directories..."
mkdir -p data
mkdir -p config
mkdir -p generated/content/blogs
mkdir -p generated/content/social
mkdir -p generated/analytics
mkdir -p generated/news
mkdir -p generated/topics

echo
echo "Setting execute permissions..."
chmod +x install.sh
chmod +x marketing_engine.py
chmod +x setup.py
chmod +x app.py

echo
echo "========================================"
echo "   Installation Complete!"
echo "========================================"
echo
echo "Next steps:"
echo "1. Add your .env file with API keys"
echo "2. Place your business document in the 'data' folder"
echo "3. Run: python setup.py (for CLI setup)"
echo "   OR"
echo "   Run: python app.py (for web dashboard)"
echo
echo "Web Dashboard: http://localhost:5000"
echo
#!/bin/bash

# Medical ML Modeling Platform - Startup Script
# Author: Rahimov M.A.

echo "🧬 Medical ML Modeling Platform"
echo "================================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "⚠️  Virtual environment not found. Creating..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Install/update dependencies
echo "📦 Checking dependencies..."
pip install -q -r requirements.txt
echo "✅ Dependencies ready"

# Run Streamlit app
echo ""
echo "🚀 Starting application..."
echo "📍 Local: http://localhost:8501"
echo "🌐 Network: Check terminal output"
echo ""
streamlit run app.py

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import json
import os
from datetime import datetime
import pandas as pd

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend to communicate

# File paths
JSON_FILE = 'waitlist_data.json'
EXCEL_FILE = 'waitlist_data.xlsx'

# Initialize JSON file if it doesn't exist
if not os.path.exists(JSON_FILE):
    with open(JSON_FILE, 'w') as f:
        json.dump([], f)

def load_waitlist_data():
    """Load existing waitlist data from JSON file"""
    try:
        with open(JSON_FILE, 'r') as f:
            return json.load(f)
    except:
        return []

def save_to_json(data):
    """Save data to JSON file"""
    with open(JSON_FILE, 'w') as f:
        json.dump(data, f, indent=2)

def convert_to_excel(data):
    """Convert JSON data to Excel file"""
    if data:
        df = pd.DataFrame(data)
        # Reorder columns for better readability
        columns = ['timestamp', 'name', 'email', 'company']
        df = df[columns]
        df.to_excel(EXCEL_FILE, index=False, sheet_name='Waitlist')
        print(f"✅ Excel file updated: {len(data)} entries")
    else:
        # Create empty Excel file
        df = pd.DataFrame(columns=['timestamp', 'name', 'email', 'company'])
        df.to_excel(EXCEL_FILE, index=False, sheet_name='Waitlist')

@app.route('/api/waitlist', methods=['POST'])
def add_to_waitlist():
    """Handle waitlist form submission"""
    try:
        # Get data from request
        data = request.json
        
        # Validate required fields
        if not data.get('name') or not data.get('email') or not data.get('company'):
            return jsonify({
                'success': False,
                'error': 'Missing required fields'
            }), 400
        
        # Load existing data
        waitlist = load_waitlist_data()
        
        # Check if email already exists
        existing_emails = [entry['email'] for entry in waitlist]
        if data['email'] in existing_emails:
            return jsonify({
                'success': False,
                'error': 'Email already registered'
            }), 400
        
        # Add timestamp
        entry = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'name': data['name'],
            'email': data['email'],
            'company': data['company']
        }
        
        # Add to waitlist
        waitlist.append(entry)
        
        # Save to JSON
        save_to_json(waitlist)
        
        # Convert to Excel
        convert_to_excel(waitlist)
        
        print(f"✅ New signup: {entry['name']} ({entry['email']})")
        
        return jsonify({
            'success': True,
            'message': 'Successfully added to waitlist',
            'total_signups': len(waitlist)
        }), 200
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/waitlist', methods=['GET'])
def get_waitlist():
    """Get all waitlist entries (for admin view)"""
    try:
        waitlist = load_waitlist_data()
        return jsonify({
            'success': True,
            'data': waitlist,
            'total': len(waitlist)
        }), 200
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/download-excel', methods=['GET'])
def download_excel():
    """Download the Excel file"""
    try:
        if os.path.exists(EXCEL_FILE):
            return send_file(
                EXCEL_FILE,
                mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                as_attachment=True,
                download_name=f'waitlist_{datetime.now().strftime("%Y%m%d")}.xlsx'
            )
        else:
            return jsonify({
                'success': False,
                'error': 'Excel file not found'
            }), 404
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/download-json', methods=['GET'])
def download_json():
    """Download the JSON file"""
    try:
        if os.path.exists(JSON_FILE):
            return send_file(
                JSON_FILE,
                mimetype='application/json',
                as_attachment=True,
                download_name=f'waitlist_{datetime.now().strftime("%Y%m%d")}.json'
            )
        else:
            return jsonify({
                'success': False,
                'error': 'JSON file not found'
            }), 404
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get waitlist statistics"""
    try:
        waitlist = load_waitlist_data()
        return jsonify({
            'success': True,
            'stats': {
                'total_signups': len(waitlist),
                'latest_signup': waitlist[-1] if waitlist else None,
                'json_file_size': os.path.getsize(JSON_FILE) if os.path.exists(JSON_FILE) else 0,
                'excel_file_size': os.path.getsize(EXCEL_FILE) if os.path.exists(EXCEL_FILE) else 0
            }
        }), 200
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/', methods=['GET'])
def home():
    """Health check endpoint"""
    return jsonify({
        'status': 'running',
        'message': '1SYX Waitlist API',
        'endpoints': {
            'POST /api/waitlist': 'Add to waitlist',
            'GET /api/waitlist': 'Get all entries',
            'GET /api/download-excel': 'Download Excel file',
            'GET /api/download-json': 'Download JSON file',
            'GET /api/stats': 'Get statistics'
        }
    })

if __name__ == '__main__':
    print("🚀 Starting 1SYX Waitlist API Server...")
    print("📊 JSON file:", JSON_FILE)
    print("📈 Excel file:", EXCEL_FILE)
    print("🌐 Server running on http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)

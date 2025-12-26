# 1SYX Waitlist Backend API

Python Flask backend for managing waitlist form submissions with automatic JSON and Excel file generation.

## Features

✅ Store form submissions in JSON format
✅ Automatically convert to Excel (.xlsx)
✅ Prevent duplicate email registrations
✅ Download Excel/JSON files
✅ View statistics and all entries
✅ CORS enabled for frontend integration

## Setup Instructions

### 1. Install Python Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Run the Server

```bash
python app.py
```

Server will start on: `http://localhost:5000`

## API Endpoints

### POST /api/waitlist
Add a new entry to the waitlist

**Request Body:**
```json
{
  "name": "John Doe",
  "email": "john@example.com",
  "company": "Acme Inc"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Successfully added to waitlist",
  "total_signups": 42
}
```

### GET /api/waitlist
Get all waitlist entries (admin view)

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "timestamp": "2025-11-30 14:07:00",
      "name": "John Doe",
      "email": "john@example.com",
      "company": "Acme Inc"
    }
  ],
  "total": 1
}
```

### GET /api/download-excel
Download the Excel file

Returns: `waitlist_YYYYMMDD.xlsx`

### GET /api/download-json
Download the JSON file

Returns: `waitlist_YYYYMMDD.json`

### GET /api/stats
Get waitlist statistics

**Response:**
```json
{
  "success": true,
  "stats": {
    "total_signups": 42,
    "latest_signup": {...},
    "json_file_size": 1024,
    "excel_file_size": 2048
  }
}
```

## Files Generated

- `waitlist_data.json` - All form submissions in JSON format
- `waitlist_data.xlsx` - Excel file with all submissions

## Excel File Format

| timestamp | name | email | company |
|-----------|------|-------|---------|
| 2025-11-30 14:07:00 | John Doe | john@example.com | Acme Inc |

## Deployment

### Option 1: Railway
```bash
# Install Railway CLI
npm i -g @railway/cli

# Login and deploy
railway login
railway init
railway up
```

### Option 2: Render
1. Create account on render.com
2. New Web Service
3. Connect your repo
4. Build command: `pip install -r requirements.txt`
5. Start command: `python app.py`

### Option 3: PythonAnywhere
1. Upload files to PythonAnywhere
2. Set up virtual environment
3. Configure WSGI file
4. Reload web app

## Environment Variables (for production)

```bash
FLASK_ENV=production
PORT=5000
```

## Security Notes

- Add authentication for admin endpoints in production
- Use environment variables for sensitive data
- Enable HTTPS in production
- Add rate limiting for form submissions

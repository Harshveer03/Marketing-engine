from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import os
import json
import threading
from datetime import datetime
from werkzeug.utils import secure_filename

# Import your existing modules
from blog_generator import BlogGenerator
from post_generator import ContentPipeline
from trend_fetcher import TrendFetcher
from performance_fetcher import collect_metrics
from performance_analyzer import LLMPerformanceAnalyzer
from feedback_loop import FeedbackLoop

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'
app.config['UPLOAD_FOLDER'] = './data'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global variable to track setup status
setup_in_progress = False

def is_configured():
    # Check if we have the essential files - prioritize existing data
    if os.path.exists("./niche/niche_icp.json") and os.path.exists("./vectordb"):
        return True
    if os.path.exists("./generated/niche_icp.json") and os.path.exists("./vectordb"):
        return True
    return os.path.exists("./config/is_configured.flag")

def load_config():
    if os.path.exists("./config/settings.json"):
        try:
            with open("./config/settings.json", "r") as f:
                return json.load(f)
        except:
            pass
    
    # Try to load from existing niche data (original location first)
    niche_file = None
    if os.path.exists("./niche/niche_icp.json"):
        niche_file = "./niche/niche_icp.json"
    elif os.path.exists("./generated/niche_icp.json"):
        niche_file = "./generated/niche_icp.json"
    
    if niche_file:
        try:
            with open(niche_file, "r") as f:
                niche_data = json.load(f)
            
            config = {
                "business_name": niche_data.get("industry", "Your Business"),
                "setup_date": datetime.now().isoformat(),
                "document_processed": "existing",
                "target_audience": niche_data.get("target_audience", []),
                "value_proposition": niche_data.get("value_proposition", "")
            }
            
            # Save the config
            os.makedirs("./config", exist_ok=True)
            with open("./config/settings.json", "w") as f:
                json.dump(config, f, indent=2)
            
            # Mark as configured
            with open("./config/is_configured.flag", "w") as f:
                f.write("configured")
            
            return config
        except Exception as e:
            print(f"Error loading niche data: {e}")
    
    return {
        "business_name": "Your Business",
        "setup_date": datetime.now().isoformat(),
        "document_processed": "existing",
        "target_audience": [],
        "value_proposition": ""
    }

def load_content_stats():
    """Load content generation statistics"""
    stats = {
        "blogs": 0,
        "social_posts": 0,
        "trends": 0,
        "last_generated": None
    }
    
    # Count blogs - check both locations
    blog_files = ["./generated/content/blogs/blogs.json", "./content/blogs/blogs.json"]
    for blog_file in blog_files:
        if os.path.exists(blog_file):
            try:
                with open(blog_file, "r", encoding="utf-8") as f:
                    blogs = json.load(f)
                    stats["blogs"] = len(blogs) if isinstance(blogs, list) else 1
                    print(f"📊 Found {stats['blogs']} blogs in {blog_file}")
                break
            except Exception as e:
                print(f"Error reading {blog_file}: {e}")
                pass
    
    # Count social posts - check both locations
    social_dirs = ["./generated/content/social", "./content/generated_content"]
    social_files = ["linkedin.json", "twitter.json", "youtube.json"]
    
    for social_dir in social_dirs:
        for platform in social_files:
            file_path = f"{social_dir}/{platform}"
            if os.path.exists(file_path):
                try:
                    with open(file_path, "r") as f:
                        posts = json.load(f)
                        if isinstance(posts, list):
                            stats["social_posts"] += len(posts)
                        else:
                            stats["social_posts"] += 1
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                    pass
    
    # Count trends - check both locations
    trends_files = ["./generated/news/filtered_news.json", "./news/filtered_news.json"]
    for trends_file in trends_files:
        if os.path.exists(trends_file):
            try:
                with open(trends_file, "r") as f:
                    trends = json.load(f)
                    stats["trends"] = len(trends) if isinstance(trends, list) else 0
                break
            except:
                pass
    
    return stats

def load_performance_data():
    """Load performance insights"""
    perf_files = ["./generated/analytics/performance_insights.json", "./analytics/performance_insights.json"]
    for perf_file in perf_files:
        if os.path.exists(perf_file):
            try:
                with open(perf_file, "r") as f:
                    return json.load(f)
            except:
                pass
    return None

@app.route('/')
def dashboard():
    if not is_configured():
        return redirect(url_for('setup'))
    
    config = load_config()
    stats = load_content_stats()
    performance = load_performance_data()
    
    return render_template('dashboard.html', 
                         config=config, 
                         stats=stats, 
                         performance=performance)

@app.route('/setup')
def setup():
    if is_configured():
        return redirect(url_for('dashboard'))
    return render_template('setup.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global setup_in_progress
    
    if setup_in_progress:
        return jsonify({'error': 'Setup already in progress'}), 400
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file selected'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        # Clear existing files in data directory
        data_dir = app.config['UPLOAD_FOLDER']
        os.makedirs(data_dir, exist_ok=True)
        
        for existing_file in os.listdir(data_dir):
            if existing_file.endswith(('.pdf', '.docx', '.txt')):
                os.remove(os.path.join(data_dir, existing_file))
        
        # Save new file
        filename = secure_filename(file.filename)
        file_path = os.path.join(data_dir, filename)
        file.save(file_path)
        
        # Run setup in background
        setup_in_progress = True
        thread = threading.Thread(target=run_setup_background)
        thread.start()
        
        return jsonify({'success': True, 'message': 'File uploaded successfully. Processing...'})
    
    return jsonify({'error': 'Invalid file type. Please upload PDF, DOCX, or TXT files.'}), 400

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'pdf', 'docx', 'txt'}

def run_setup_background():
    global setup_in_progress
    try:
        # Import and run setup
        from simple_embedder import create_embeddings
        from extractor import extract_structured_info, load_faiss_index, query_icp
        
        # Create embeddings
        create_embeddings()
        
        # Extract niche
        vectordb = load_faiss_index()
        context = query_icp(vectordb)
        niche_data = extract_structured_info(context)
        
        # Save niche data
        os.makedirs("./generated", exist_ok=True)
        with open("./generated/niche_icp.json", "w", encoding="utf-8") as f:
            json.dump(niche_data, f, indent=4, ensure_ascii=False)
        
        # Mark as configured
        os.makedirs("./config", exist_ok=True)
        with open("./config/is_configured.flag", "w") as f:
            f.write("configured")
            
    except Exception as e:
        print(f"Setup error: {e}")
    finally:
        setup_in_progress = False

@app.route('/setup_status')
def setup_status():
    return jsonify({
        'in_progress': setup_in_progress,
        'configured': is_configured()
    })

@app.route('/generate/<content_type>')
def generate_content(content_type):
    if not is_configured():
        return jsonify({'error': 'System not configured'}), 400
    
    try:
        if content_type == 'blog':
            generator = BlogGenerator()
            result = generator.run(mode="automatic")
            print(f"📝 Blog generation result: {result}")
            
            # Verify the file was created
            blog_file = "./generated/content/blogs/blogs.json"
            if os.path.exists(blog_file):
                with open(blog_file, "r") as f:
                    blogs = json.load(f)
                    print(f"✅ Blog file exists with {len(blogs)} blogs")
            else:
                print(f"❌ Blog file not found at {blog_file}")
            
            return jsonify({'success': True, 'title': result.get('title', 'Blog Generated')})
        
        elif content_type == 'social':
            pipeline = ContentPipeline()
            result = pipeline.run()
            print(f"Social content generation result: {result}")
            return jsonify({'success': True, 'message': 'Social content generated for all platforms'})
        
        elif content_type == 'trends':
            fetcher = TrendFetcher()
            results = fetcher.run("./generated/niche_icp.json")
            return jsonify({'success': True, 'count': len(results)})
        
        elif content_type == 'analysis':
            # Run performance analysis
            collect_metrics()
            analyzer = LLMPerformanceAnalyzer()
            analyzer.run()
            feedback = FeedbackLoop()
            feedback.run()
            return jsonify({'success': True, 'message': 'Performance analysis complete'})
        
        else:
            return jsonify({'error': 'Invalid content type'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/content/<content_type>')
def view_content(content_type):
    """API endpoint to get content data"""
    try:
        if content_type == 'blogs':
            # Check both locations
            file_paths = ["./generated/content/blogs/blogs.json", "./content/blogs/blogs.json"]
            for file_path in file_paths:
                if os.path.exists(file_path):
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    return jsonify(data)
            return jsonify([])
            
        elif content_type == 'social':
            # Return all social content - check both locations
            social_content = {}
            social_dirs = ["./generated/content/social", "./content/generated_content"]
            
            for platform in ['linkedin', 'twitter', 'youtube']:
                for social_dir in social_dirs:
                    file_path = f"{social_dir}/{platform}.json"
                    if os.path.exists(file_path):
                        with open(file_path, "r", encoding="utf-8") as f:
                            social_content[platform] = json.load(f)
                        break
            return jsonify(social_content)
            
        elif content_type == 'trends':
            # Check both locations
            file_paths = ["./generated/news/filtered_news.json", "./news/filtered_news.json"]
            for file_path in file_paths:
                if os.path.exists(file_path):
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    return jsonify(data)
            return jsonify([])
            
        elif content_type == 'performance':
            # Check both locations
            file_paths = ["./generated/analytics/performance_insights.json", "./analytics/performance_insights.json"]
            for file_path in file_paths:
                if os.path.exists(file_path):
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    return jsonify(data)
            return jsonify([])
        else:
            return jsonify({'error': 'Invalid content type'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/export')
def export_content():
    """Export all content as JSON"""
    if not is_configured():
        return jsonify({'error': 'System not configured'}), 400
    
    try:
        config = load_config()
        export_data = {
            "config": config,
            "export_date": datetime.now().isoformat(),
            "content": {}
        }
        
        # Export all content
        content_files = [
            ("blogs", "./generated/content/blogs/blogs.json"),
            ("social_linkedin", "./generated/content/social/linkedin.json"),
            ("social_twitter", "./generated/content/social/twitter.json"), 
            ("social_youtube", "./generated/content/social/youtube.json"),
            ("trends", "./generated/news/filtered_news.json"),
            ("analytics", "./generated/analytics/performance_insights.json")
        ]
        
        for content_type, file_path in content_files:
            if os.path.exists(file_path):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        export_data["content"][content_type] = json.load(f)
                except:
                    pass
        
        return jsonify(export_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stats')
def get_stats():
    """Get current content statistics"""
    try:
        stats = load_content_stats()
        print(f"📊 Stats endpoint called - returning: {stats}")
        return jsonify(stats)
    except Exception as e:
        print(f"❌ Error in stats endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/generate_topics')
def generate_topics():
    """Generate topics for social content selection"""
    if not is_configured():
        return jsonify({'error': 'System not configured'}), 400
    
    try:
        from post_generator import ContentPipeline
        
        # Create pipeline instance
        pipeline = ContentPipeline()
        
        # Call the generate_topics method directly (not run())
        topics = pipeline.generate_topics()
        
        # Topics generated successfully for web UI
        
        return jsonify({'success': True, 'topics': topics})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/generate_social_with_selection', methods=['POST'])
def generate_social_with_selection():
    """Generate social content with user selections"""
    if not is_configured():
        return jsonify({'error': 'System not configured'}), 400
    
    try:
        data = request.get_json()
        topic_index = data.get('topic_index')
        tone = data.get('tone', 'professional')
        audience = data.get('audience', 'Founders')
        
        if topic_index is None:
            return jsonify({'error': 'Missing topic selection'}), 400
        
        from post_generator import ContentPipeline
        pipeline = ContentPipeline()
        
        # Generate topics to get the selected one
        topics = pipeline.generate_topics()
        if topic_index >= len(topics):
            return jsonify({'error': 'Invalid topic index'}), 400
        
        selected_topic = topics[topic_index]
        
        # Save the selected topic
        pipeline.save_json("./generated/topics/topics.json", [{"title": selected_topic["title"], "related_news": selected_topic.get("related_news", [])}])
        
        # Get context and generate content
        niche, pdf_context = pipeline.get_context(selected_topic)
        
        # Generate content for all platforms
        linkedin = pipeline.generate_linkedin(selected_topic, selected_topic.get("related_news", []), niche, audience, tone, pdf_context)
        twitter = pipeline.generate_twitter(selected_topic, selected_topic.get("related_news", []), niche, audience, tone, pdf_context)
        youtube = pipeline.generate_youtube(selected_topic, selected_topic.get("related_news", []), niche, audience, tone, pdf_context)
        
        # Save content to files
        import os
        output_dir = "./generated/content/social"
        os.makedirs(output_dir, exist_ok=True)
        
        linkedin_data = {
            "title": selected_topic["title"],
            "caption": linkedin.get("caption", ""),
            "hashtags": linkedin.get("hashtags", [])
        }
        
        twitter_data = {
            "title": selected_topic["title"],
            "caption": twitter.get("tweet", ""),
            "hashtags": twitter.get("hashtags", [])
        }
        
        youtube_data = {
            "title": selected_topic["title"],
            "script_intro": youtube.get("script_intro", ""),
            "caption": youtube.get("description", ""),
            "hashtags": youtube.get("tags", [])
        }
        
        pipeline.append_json(os.path.join(output_dir, "linkedin.json"), linkedin_data)
        pipeline.append_json(os.path.join(output_dir, "twitter.json"), twitter_data)
        pipeline.append_json(os.path.join(output_dir, "youtube.json"), youtube_data)
        
        return jsonify({'success': True, 'message': 'Social content generated successfully'})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
        return jsonify({'error': str(e)}), 500

@app.route('/reset')
def reset_system():
    """Reset the system configuration"""
    try:
        # Remove configuration files
        if os.path.exists("./config/is_configured.flag"):
            os.remove("./config/is_configured.flag")
        if os.path.exists("./config/settings.json"):
            os.remove("./config/settings.json")
        
        flash('System reset successfully. Please upload a new document to reconfigure.')
        return redirect(url_for('setup'))
    except Exception as e:
        flash(f'Error resetting system: {e}')
        return redirect(url_for('dashboard'))

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('./data', exist_ok=True)
    os.makedirs('./config', exist_ok=True)
    os.makedirs('./templates', exist_ok=True)
    os.makedirs('./static/css', exist_ok=True)
    os.makedirs('./static/js', exist_ok=True)
    
    print("🚀 Marketing Engine Dashboard Starting...")
    print("📊 Open your browser to: http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
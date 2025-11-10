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
                    content = f.read().strip()
                    if content:  # Check if file has content
                        blogs = json.loads(content)
                        stats["blogs"] = len(blogs) if isinstance(blogs, list) else 1
                        print(f"📊 Found {stats['blogs']} blogs in {blog_file}")
                    else:
                        print(f"📊 Empty blog file: {blog_file}")
                break
            except json.JSONDecodeError as e:
                print(f"Error reading {blog_file}: {e}")
                # Initialize empty file
                with open(blog_file, "w", encoding="utf-8") as f:
                    json.dump([], f)
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
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                        if content:  # Check if file has content
                            posts = json.loads(content)
                            if isinstance(posts, list):
                                stats["social_posts"] += len(posts)
                            else:
                                stats["social_posts"] += 1
                        else:
                            print(f"📊 Empty social file: {file_path}")
                except json.JSONDecodeError as e:
                    print(f"Error reading {file_path}: {e}")
                    # Initialize empty file
                    with open(file_path, "w", encoding="utf-8") as f:
                        json.dump([], f)
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
        
        # Store topics for later use in content generation
        stored_topics_file = "./generated/topics/current_session_topics.json"
        os.makedirs(os.path.dirname(stored_topics_file), exist_ok=True)
        with open(stored_topics_file, "w", encoding="utf-8") as f:
            json.dump(topics, f, indent=2, ensure_ascii=False)
        
        print(f"🎯 Generated and stored {len(topics)} topics for user selection")
        for i, topic in enumerate(topics):
            print(f"  {i}: {topic['title']}")
        
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
        print(f"📥 Received request data: {data}")
        
        topic_index = data.get('topic_index')
        industry = data.get('industry', 'IT & Dev')
        tone = data.get('tone', 'professional')
        audience = data.get('audience', 'Founders')
        
        print(f"🔢 Topic index: {topic_index}, Industry: {industry}, Tone: {tone}, Audience: {audience}")
        
        if topic_index is None:
            return jsonify({'error': 'Missing topic selection'}), 400
        
        from post_generator import ContentPipeline
        pipeline = ContentPipeline()
        
        # Load the previously generated topics from session storage
        stored_topics_file = "./generated/topics/current_session_topics.json"
        if os.path.exists(stored_topics_file):
            with open(stored_topics_file, "r", encoding="utf-8") as f:
                topics = json.load(f)
            print(f"📂 Loaded {len(topics)} topics from session storage")
            for i, topic in enumerate(topics):
                print(f"  {i}: {topic['title']}")
        else:
            print(f"❌ Session topics file not found: {stored_topics_file}")
            return jsonify({'error': 'No topics found. Please refresh and try again.'}), 400
        
        if topic_index >= len(topics):
            print(f"❌ Invalid topic index {topic_index}, only {len(topics)} topics available")
            return jsonify({'error': f'Invalid topic index {topic_index}. Available: 0-{len(topics)-1}'}), 400
        
        selected_topic = topics[topic_index]
        print(f"🎯 User selected topic #{topic_index}: '{selected_topic['title']}'")
        print(f"📰 Related news items: {len(selected_topic.get('related_news', []))}")
        
        # Save the selected topic
        pipeline.save_json("./generated/topics/topics.json", [{"title": selected_topic["title"], "related_news": selected_topic.get("related_news", [])}])
        
        # Get context and generate content
        try:
            niche, pdf_context = pipeline.get_context(selected_topic)
        except Exception as e:
            print(f"Error getting context: {e}")
            return jsonify({'error': f'Failed to get context: {str(e)}'}), 500
        
        # Generate content for all platforms using the selected topic
        print(f"🚀 Generating content for topic: '{selected_topic['title']}'")
        print(f"📊 Industry: {industry}, Tone: {tone}, Audience: {audience}")
        
        linkedin = pipeline.generate_linkedin(selected_topic, selected_topic.get("related_news", []), niche, audience, tone, pdf_context, industry)
        twitter = pipeline.generate_twitter(selected_topic, selected_topic.get("related_news", []), niche, audience, tone, pdf_context, industry)
        youtube = pipeline.generate_youtube(selected_topic, selected_topic.get("related_news", []), niche, audience, tone, pdf_context, industry)
        
        # Calculate quality scores for each platform
        linkedin_quality = pipeline.calculate_social_quality_score(
            "linkedin",
            selected_topic,
            linkedin.get("caption", ""),
            linkedin.get("hashtags", [])
        )
        
        twitter_quality = pipeline.calculate_social_quality_score(
            "twitter",
            selected_topic,
            twitter.get("tweet", ""),
            twitter.get("hashtags", [])
        )
        
        youtube_quality = pipeline.calculate_social_quality_score(
            "youtube",
            selected_topic,
            youtube.get("script_intro", "") + " " + youtube.get("description", ""),
            youtube.get("tags", [])
        )
        
        # Save content to files
        output_dir = "./generated/content/social"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save content with the selected topic title
        topic_title = selected_topic["title"]
        
        linkedin_data = {
            "title": topic_title,
            "caption": linkedin.get("caption", ""),
            "hashtags": linkedin.get("hashtags", []),
            "quality_score": linkedin_quality,
            "industry": industry,
            "tone": tone,
            "audience": audience,
            "timestamp": datetime.now().isoformat()
        }
        
        twitter_data = {
            "title": topic_title,
            "caption": twitter.get("tweet", ""),
            "hashtags": twitter.get("hashtags", []),
            "quality_score": twitter_quality,
            "industry": industry,
            "tone": tone,
            "audience": audience,
            "timestamp": datetime.now().isoformat()
        }
        
        youtube_data = {
            "title": topic_title,
            "script_intro": youtube.get("script_intro", ""),
            "caption": youtube.get("description", ""),
            "hashtags": youtube.get("tags", []),
            "quality_score": youtube_quality,
            "industry": industry,
            "tone": tone,
            "audience": audience,
            "timestamp": datetime.now().isoformat()
        }
        
        print(f"💾 Saving content for topic: '{topic_title}'")
        
        pipeline.append_json(os.path.join(output_dir, "linkedin.json"), linkedin_data)
        pipeline.append_json(os.path.join(output_dir, "twitter.json"), twitter_data)
        pipeline.append_json(os.path.join(output_dir, "youtube.json"), youtube_data)
        
        return jsonify({'success': True, 'message': 'Social content generated successfully'})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/generate_blog_with_selection', methods=['POST'])
def generate_blog_with_selection():
    """Generate blog content with user selections"""
    if not is_configured():
        return jsonify({'error': 'System not configured'}), 400
    
    try:
        data = request.get_json()
        print(f"📥 Received blog request data: {data}")
        
        topic_index = data.get('topic_index')
        industry = data.get('industry', 'IT & Dev')
        tone = data.get('tone', 'professional')
        audience = data.get('audience', 'Founders')
        
        print(f"🔢 Blog topic index: {topic_index}, Industry: {industry}, Tone: {tone}, Audience: {audience}")
        
        if topic_index is None:
            return jsonify({'error': 'Missing topic selection'}), 400
        
        from blog_generator import BlogGenerator
        generator = BlogGenerator()
        
        # Load the previously generated topics from session storage
        stored_topics_file = "./generated/topics/current_session_topics.json"
        if os.path.exists(stored_topics_file):
            with open(stored_topics_file, "r", encoding="utf-8") as f:
                topics = json.load(f)
            print(f"📂 Loaded {len(topics)} topics from session storage for blog")
            for i, topic in enumerate(topics):
                print(f"  {i}: {topic['title']}")
        else:
            print(f"❌ Session topics file not found: {stored_topics_file}")
            return jsonify({'error': 'No topics found. Please refresh and try again.'}), 400
        
        if topic_index >= len(topics):
            print(f"❌ Invalid topic index {topic_index}, only {len(topics)} topics available")
            return jsonify({'error': f'Invalid topic index {topic_index}. Available: 0-{len(topics)-1}'}), 400
        
        selected_topic = topics[topic_index]
        print(f"🎯 User selected blog topic #{topic_index}: '{selected_topic['title']}'")
        print(f"📰 Related news items: {len(selected_topic.get('related_news', []))}")
        
        # Generate blog content using the selected topic
        print(f"🚀 Generating blog for topic: '{selected_topic['title']}'")
        print(f"📊 Industry: {industry}, Tone: {tone}, Audience: {audience}")
        
        # Get niche data and context
        niche = generator.load_json("./generated/niche_icp.json")
        pdf_context = generator.build_pdf_context(selected_topic['title'], niche)
        
        # Generate blog with industry context
        blog_data = generator.generate_blog_with_industry(
            selected_topic['title'], 
            selected_topic.get('related_news', []), 
            niche, 
            pdf_context,
            industry,
            tone,
            audience
        )
        
        # Calculate quality score
        quality_score = generator.calculate_quality_score(
            selected_topic['title'],
            blog_data.get("blog", ""),
            selected_topic.get('related_news', [])
        )
        
        # Save blog content
        blog_entry = {
            "title": blog_data.get("title", selected_topic['title']),
            "outline": blog_data.get("outline", []),
            "blog": blog_data.get("blog", ""),
            "news": selected_topic.get('related_news', []),
            "industry": industry,
            "tone": tone,
            "audience": audience,
            "quality_score": quality_score,
            "timestamp": datetime.now().isoformat()
        }
        
        print(f"💾 Saving blog for topic: '{blog_entry['title']}'")
        generator.append_json("./generated/content/blogs/blogs.json", blog_entry)
        
        # Save the used topic to prevent duplicates
        print(f"🔄 Starting to save used topic: '{selected_topic['title']}'")
        used_topics_file = "./generated/topics/used_blog_topics.json"
        print(f"📁 Creating directory for: {used_topics_file}")
        os.makedirs(os.path.dirname(used_topics_file), exist_ok=True)
        
        used_topics = []
        if os.path.exists(used_topics_file):
            print(f"📖 Reading existing used topics from: {used_topics_file}")
            with open(used_topics_file, "r", encoding="utf-8") as f:
                try:
                    used_topics = json.load(f)
                    print(f"📊 Found {len(used_topics)} existing used topics")
                except json.JSONDecodeError as e:
                    print(f"⚠️ JSON decode error in used topics file: {e}")
                    used_topics = []
        else:
            print(f"📄 Used topics file doesn't exist, will create new one")
        
        new_topic_entry = {"title": selected_topic['title'], "generated_on": datetime.now().isoformat()}
        used_topics.append(new_topic_entry)
        print(f"➕ Adding new topic entry: {new_topic_entry}")
        
        with open(used_topics_file, "w", encoding="utf-8") as f:
            json.dump(used_topics, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Successfully saved used topic: '{selected_topic['title']}' to {used_topics_file}")
        print(f"📊 Total used topics now: {len(used_topics)}")
        
        return jsonify({'success': True, 'message': 'Blog content generated successfully', 'title': blog_entry['title']})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate_image_prompt', methods=['POST'])
def generate_image_prompt():
    """Generate optimized image prompt for blog, LinkedIn, or Twitter content, or video prompt for YouTube"""
    try:
        data = request.get_json()
        content_type = data.get('type', 'blog')
        content_data = data.get('data', {})
        
        print(f"📸 Generating prompt for {content_type}")
        print(f"🔍 DEBUG - content_type value: '{content_type}' (type: {type(content_type).__name__})")
        print(f"🔍 DEBUG - Full request data: {data}")
        print(f"📝 Content title: {content_data.get('title', 'N/A')}")
        
        if content_type == 'blog':
            from image_prompt_builder import blog_image_prompt
            prompt = blog_image_prompt(content_data)
            
            print(f"✅ Blog image prompt generated successfully")
            return jsonify({
                'success': True, 
                'prompt': prompt,
                'message': 'Image prompt generated successfully'
            })
        elif content_type == 'linkedin':
            from image_prompt_builder import linkedin_image_prompt
            prompt = linkedin_image_prompt(content_data)
            
            print(f"✅ LinkedIn image prompt generated successfully")
            return jsonify({
                'success': True, 
                'prompt': prompt,
                'message': 'LinkedIn image prompt generated successfully'
            })
        elif content_type == 'twitter':
            from image_prompt_builder import twitter_image_prompt
            prompt = twitter_image_prompt(content_data)
            
            print(f"✅ Twitter image prompt generated successfully")
            return jsonify({
                'success': True, 
                'prompt': prompt,
                'message': 'Twitter image prompt generated successfully'
            })
        elif content_type == 'youtube':
            from image_prompt_builder import youtube_video_prompt
            prompt = youtube_video_prompt(content_data)
            
            print(f"✅ YouTube video prompt generated successfully")
            return jsonify({
                'success': True, 
                'prompt': prompt,
                'message': 'YouTube video prompt generated successfully'
            })
        else:
            return jsonify({'error': 'Invalid content type'}), 400
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ Error generating prompt: {e}")
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
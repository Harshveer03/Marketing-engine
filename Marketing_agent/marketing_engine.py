import os
import json
from datetime import datetime

# Import your existing modules
from blog_generator import BlogGenerator
from post_generator import ContentPipeline
from trend_fetcher import TrendFetcher
from performance_fetcher import collect_metrics
from performance_analyzer import LLMPerformanceAnalyzer
from feedback_loop import FeedbackLoop

class MarketingEngine:
    def __init__(self):
        self.check_setup()
        self.load_config()
        self.setup_paths()
    
    def check_setup(self):
        if not os.path.exists("./config/is_configured.flag"):
            print("❌ Marketing Engine not configured yet!")
            print("🔧 Run 'python setup.py' first to set up your marketing engine.")
            print("📊 Or run 'python app.py' for the web dashboard setup.")
            exit(1)
    
    def load_config(self):
        try:
            with open("./config/settings.json", "r") as f:
                self.config = json.load(f)
        except FileNotFoundError:
            print("❌ Configuration file not found!")
            print("🔧 Run 'python setup.py' to reconfigure.")
            exit(1)
    
    def setup_paths(self):
        """Ensure all necessary directories exist"""
        directories = [
            "./generated/content/blogs",
            "./generated/content/social", 
            "./generated/analytics",
            "./generated/news",
            "./generated/topics"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def run(self):
        print(f"🚀 {self.config['business_name']} Marketing Engine")
        print("=" * 60)
        print(f"📅 Configured: {self.config['setup_date'][:10]}")
        print(f"🎯 Industry: {self.config['business_name']}")
        
        while True:
            self.show_menu()
            choice = input("\n👉 Select option: ").strip()
            
            if choice == "1":
                self.generate_blog()
            elif choice == "2":
                self.generate_social_content()
            elif choice == "3":
                self.fetch_trends()
            elif choice == "4":
                self.run_performance_analysis()
            elif choice == "5":
                self.view_analytics()
            elif choice == "6":
                self.export_content()
            elif choice == "7":
                self.show_config()
            elif choice == "8":
                print("👋 Goodbye! Your content is saved in ./generated/")
                break
            else:
                print("❌ Invalid choice! Please select 1-8.")
    
    def show_menu(self):
        print("\n" + "="*60)
        print("📋 What would you like to do?")
        print("1. 📝 Generate Blog Post")
        print("2. 📱 Generate Social Media Content") 
        print("3. 📰 Fetch Latest Industry Trends")
        print("4. 📊 Run Performance Analysis")
        print("5. 📈 View Analytics Dashboard")
        print("6. 💾 Export All Content")
        print("7. ⚙️  View Configuration")
        print("8. 🚪 Exit")
    
    def generate_blog(self):
        print("\n🔄 Generating blog post...")
        try:
            # Update paths for simplified structure
            generator = BlogGenerator()
            blog = generator.run(mode="automatic")
            print(f"✅ Blog generated: {blog['title']}")
            print(f"📁 Saved to: ./generated/content/blogs/")
        except Exception as e:
            print(f"❌ Error generating blog: {e}")
    
    def generate_social_content(self):
        print("\n🔄 Generating social media content...")
        try:
            pipeline = ContentPipeline()
            pipeline.run()
            print("✅ Social content generated for all platforms!")
            print("📁 Saved to: ./generated/content/social/")
        except Exception as e:
            print(f"❌ Error generating social content: {e}")
    
    def fetch_trends(self):
        print("\n🔄 Fetching latest industry trends...")
        try:
            fetcher = TrendFetcher()
            results = fetcher.run("./generated/niche_icp.json")
            print(f"✅ Found {len(results)} relevant trends")
            print("📁 Saved to: ./generated/news/")
        except Exception as e:
            print(f"❌ Error fetching trends: {e}")
    
    def run_performance_analysis(self):
        print("\n🔄 Running performance analysis...")
        try:
            # Collect metrics
            collect_metrics()
            
            # Analyze performance
            analyzer = LLMPerformanceAnalyzer()
            analyzer.run()
            
            # Generate feedback
            feedback = FeedbackLoop()
            feedback.run()
            
            print("✅ Performance analysis complete!")
            print("📁 Results saved to: ./generated/analytics/")
        except Exception as e:
            print(f"❌ Error in performance analysis: {e}")
    
    def view_analytics(self):
        print("\n📊 Analytics Overview")
        print("-" * 40)
        
        try:
            # Show content stats
            self.show_content_stats()
            
            # Show performance data if available
            perf_file = "./generated/analytics/performance_insights.json"
            if os.path.exists(perf_file):
                with open(perf_file, "r") as f:
                    insights = json.load(f)
                    self.display_performance_summary(insights)
            else:
                print("📈 No performance data yet. Run option 4 to generate analytics.")
                
        except Exception as e:
            print(f"❌ Error loading analytics: {e}")
    
    def show_content_stats(self):
        """Display content generation statistics"""
        stats = {
            "blogs": 0,
            "social_posts": 0,
            "trends": 0
        }
        
        # Count blogs
        blog_file = "./generated/content/blogs/blogs.json"
        if os.path.exists(blog_file):
            try:
                with open(blog_file, "r") as f:
                    blogs = json.load(f)
                    stats["blogs"] = len(blogs) if isinstance(blogs, list) else 1
            except:
                pass
        
        # Count social posts
        social_files = ["linkedin.json", "twitter.json", "youtube.json"]
        for platform in social_files:
            file_path = f"./generated/content/social/{platform}"
            if os.path.exists(file_path):
                try:
                    with open(file_path, "r") as f:
                        posts = json.load(f)
                        stats["social_posts"] += len(posts) if isinstance(posts, list) else 1
                except:
                    pass
        
        # Count trends
        trends_file = "./generated/news/filtered_news.json"
        if os.path.exists(trends_file):
            try:
                with open(trends_file, "r") as f:
                    trends = json.load(f)
                    stats["trends"] = len(trends) if isinstance(trends, list) else 0
            except:
                pass
        
        print(f"📝 Blog Posts: {stats['blogs']}")
        print(f"📱 Social Posts: {stats['social_posts']}")
        print(f"📰 Trends Tracked: {stats['trends']}")
    
    def display_performance_summary(self, insights):
        """Display key performance insights"""
        print("\n🎯 Performance Insights:")
        
        try:
            platforms = insights.get("summary", {}).get("platforms", {})
            for platform, data in platforms.items():
                engagement = data.get("avg_engagement", 0)
                print(f"  {platform.capitalize()}: {engagement:.1%} avg engagement")
            
            global_insights = insights.get("summary", {}).get("global_insights", {})
            recommendation = global_insights.get("overall_recommendation", "")
            if recommendation:
                print(f"\n💡 Key Recommendation:")
                print(f"  {recommendation[:100]}...")
                
        except Exception as e:
            print(f"❌ Error displaying performance data: {e}")
    
    def export_content(self):
        print("\n💾 Exporting all content...")
        
        export_data = {
            "config": self.config,
            "export_date": datetime.now().isoformat(),
            "content": {}
        }
        
        # Export all generated content
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
                except Exception as e:
                    print(f"⚠️ Could not export {content_type}: {e}")
        
        # Save export
        export_file = f"./generated/content_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(export_file, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Content exported to: {export_file}")
    
    def show_config(self):
        print("\n⚙️ Current Configuration:")
        print("-" * 40)
        for key, value in self.config.items():
            if isinstance(value, list):
                print(f"{key}: {', '.join(value)}")
            else:
                print(f"{key}: {value}")

if __name__ == "__main__":
    engine = MarketingEngine()
    engine.run()
import os
import requests
from dotenv import load_dotenv

load_dotenv()

class PexelsHelper:
    """
    Helper class to interact with Pexels API for fetching images
    """
    
    def __init__(self):
        self.api_key = os.getenv("PEXELS_API_KEY")
        self.base_url = "https://api.pexels.com/v1"
        
        if not self.api_key:
            print("⚠️ PEXELS_API_KEY not found in .env file")
    
    def search_images(self, query, per_page=5, orientation=None):
        """
        Search for images by keyword
        
        Args:
            query (str): Search query
            per_page (int): Number of results (max 80)
            orientation (str): 'landscape', 'portrait', or 'square'
        
        Returns:
            list: List of image objects
        """
        if not self.api_key:
            print("❌ Pexels API key not configured")
            return []
        
        headers = {"Authorization": self.api_key}
        params = {
            "query": query,
            "per_page": per_page
        }
        
        if orientation:
            params["orientation"] = orientation
        
        try:
            response = requests.get(
                f"{self.base_url}/search",
                headers=headers,
                params=params,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"🖼️ Found {len(data.get('photos', []))} images for '{query}'")
                return data.get("photos", [])
            elif response.status_code == 429:
                print("⚠️ Pexels API rate limit exceeded")
                return []
            else:
                print(f"❌ Pexels API error: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"❌ Error fetching images from Pexels: {e}")
            return []
    
    def get_best_image(self, query, orientation=None, quality="high"):
        """
        Get the best matching image for a query
        
        Args:
            query (str): Search query
            orientation (str): 'landscape', 'portrait', or 'square'
            quality (str): 'high', 'medium', or 'low' - determines which URL to use as default
        
        Returns:
            dict: Image data with url, photographer, and alt text
        """
        images = self.search_images(query, per_page=1, orientation=orientation)
        
        if images:
            photo = images[0]
            
            # Select default URL based on quality preference
            quality_map = {
                "high": photo["src"]["original"],      # Highest quality (original size)
                "medium": photo["src"]["large2x"],     # 2x large (very good quality)
                "low": photo["src"]["large"]           # Standard large
            }
            
            default_url = quality_map.get(quality, photo["src"]["original"])
            
            return {
                "url": default_url,                    # Default URL based on quality setting
                "url_original": photo["src"]["original"],  # Full resolution (largest file)
                "url_large2x": photo["src"]["large2x"],    # 2x large (retina quality)
                "url_large": photo["src"]["large"],        # Large (940px wide)
                "url_medium": photo["src"]["medium"],      # Medium (350px wide)
                "url_small": photo["src"]["small"],        # Small (130px wide)
                "photographer": photo["photographer"],
                "photographer_url": photo["photographer_url"],
                "alt": photo.get("alt", query),
                "width": photo["width"],
                "height": photo["height"],
                "pexels_url": photo["url"]
            }
        
        return None
    
    def get_multiple_images(self, query, count=3, orientation=None):
        """
        Get multiple images for a query
        
        Args:
            query (str): Search query
            count (int): Number of images to return
            orientation (str): 'landscape', 'portrait', or 'square'
        
        Returns:
            list: List of image data dictionaries
        """
        images = self.search_images(query, per_page=count, orientation=orientation)
        
        results = []
        for photo in images:
            results.append({
                "url": photo["src"]["large"],
                "url_original": photo["src"]["original"],
                "url_medium": photo["src"]["medium"],
                "url_small": photo["src"]["small"],
                "photographer": photo["photographer"],
                "photographer_url": photo["photographer_url"],
                "alt": photo.get("alt", query),
                "width": photo["width"],
                "height": photo["height"],
                "pexels_url": photo["url"]
            })
        
        return results
    
    def get_image_for_blog(self, topic, content_snippet=None, quality="high"):
        """
        Get an appropriate image for a blog post
        Uses AI-powered query generation with automatic fallback
        
        Args:
            topic (str): Blog topic/title
            content_snippet (str): Optional snippet of blog content for better context
            quality (str): 'high', 'medium', or 'low'
        
        Returns:
            dict: Image data
        """
        # Try AI-powered query generation
        try:
            smart_query = self.generate_smart_query_with_ai(topic, content_snippet)
            image = self.get_best_image(smart_query, orientation="landscape", quality=quality)
            if image:
                return image
        except Exception as e:
            print(f"⚠️ AI query failed: {e}")
        
        # Fallback: Use topic directly
        return self.get_best_image(topic, orientation="landscape", quality=quality)
    
    def get_image_for_social(self, topic, content_snippet=None, platform="linkedin", quality="high"):
        """
        Get an appropriate image for social media post
        Uses AI-powered query generation with automatic fallback
        
        Args:
            topic (str): Post topic
            content_snippet (str): Optional snippet of post content for better context
            platform (str): 'linkedin', 'twitter', 'youtube'
            quality (str): 'high', 'medium', or 'low'
        
        Returns:
            dict: Image data
        """
        orientations = {
            "linkedin": "landscape",
            "twitter": "landscape",
            "youtube": "landscape",
            "instagram": "square"
        }
        orientation = orientations.get(platform, "landscape")
        
        # Try AI-powered query generation
        try:
            smart_query = self.generate_smart_query_with_ai(topic, content_snippet)
            image = self.get_best_image(smart_query, orientation=orientation, quality=quality)
            if image:
                return image
        except Exception as e:
            print(f"⚠️ AI query failed: {e}")
        
        # Fallback: Use topic directly
        return self.get_best_image(topic, orientation=orientation, quality=quality)
    
    def extract_keywords_from_topic(self, topic):
        """
        Extract searchable keywords from a topic
        
        Args:
            topic (str): Topic string
        
        Returns:
            list: List of keywords
        """
        # Remove common words
        stop_words = {
            'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'and', 'or', 'but',
            'how', 'what', 'why', 'when', 'where', 'which', 'who', 'with', 'from',
            'this', 'that', 'these', 'those', 'your', 'you', 'are', 'is', 'be'
        }
        
        words = topic.lower().split()
        keywords = [word for word in words if word not in stop_words and len(word) > 3]
        
        return keywords[:3]  # Return top 3 keywords
    
    def generate_smart_query_with_ai(self, topic, content_snippet=None):
        """
        Generate a smart search query using Gemini AI to extract visual concepts
        
        Args:
            topic (str): Content topic/title
            content_snippet (str): Optional snippet of content for context
        
        Returns:
            str: Optimized search query for Pexels
        """
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            
            llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0)
            
            context = f"Topic: {topic}"
            if content_snippet:
                context += f"\nContent: {content_snippet[:500]}"
            
            prompt = f"""
            You are an expert at finding the perfect stock photo for content. Generate a 2-4 word search query for Pexels that will find the MOST VISUALLY RELEVANT image.
            
            {context}
            
            CRITICAL RULES:
            1. Focus on CONCRETE, PHOTOGRAPHABLE subjects (people, objects, settings)
            2. Avoid abstract concepts - convert them to visual equivalents
            3. Include WHO (person/role) + WHAT (action/object) + WHERE (setting) when relevant
            4. Use common stock photo terminology
            5. Be specific enough to be relevant, but broad enough to find results
            6. Return ONLY the search query - no quotes, no explanation
            
            TRANSFORMATION EXAMPLES:
            Abstract → Concrete:
            - "success" → "business handshake celebration"
            - "innovation" → "team brainstorming whiteboard"
            - "growth" → "business chart upward trend"
            - "transformation" → "person using modern technology"
            
            Topic → Visual Query:
            - "How AI is transforming healthcare" → "doctor using digital tablet patient"
            - "Remote work productivity tips" → "professional home office workspace"
            - "Customer service best practices" → "support agent helping customer smiling"
            - "Data analytics for startups" → "business team analyzing dashboard screen"
            - "Leadership in tech companies" → "business leader presenting team meeting"
            - "Cybersecurity threats 2025" → "security professional monitoring computer screens"
            - "Sustainable business practices" → "business team green office environment"
            
            INDUSTRY-SPECIFIC PATTERNS:
            - Healthcare: "medical professional" + action/technology
            - Technology: "professional" + device/screen + setting
            - Finance: "business" + financial activity + professional setting
            - Education: "teacher/student" + learning activity + classroom
            - Retail: "customer" + shopping activity + store
            
            Now generate the search query:
            """
            
            response = llm.invoke(prompt).content.strip()
            
            # Clean up the response
            query = response.replace('"', '').replace("'", "").strip()
            
            # Remove common prefixes/suffixes that AI might add
            prefixes_to_remove = [
                'search query:', 'query:', 'search:', 'image search:',
                'pexels search:', 'stock photo:', 'photo of', 'image of'
            ]
            for prefix in prefixes_to_remove:
                if query.lower().startswith(prefix):
                    query = query[len(prefix):].strip()
            
            # Remove trailing punctuation
            query = query.rstrip('.,!?;:')
            
            # Validate query length (2-6 words is ideal for Pexels)
            words = query.split()
            if len(words) > 6:
                # Too long, take first 6 words
                query = ' '.join(words[:6])
                print(f"⚠️ Query too long, truncated to: '{query}'")
            elif len(words) < 2:
                # Too short, fall back to rule-based
                print(f"⚠️ AI query too short: '{query}', using fallback")
                return self.generate_smart_query_rule_based(topic, content_snippet)
            
            # Validate query doesn't contain unwanted characters
            if any(char in query for char in ['[', ']', '{', '}', '(', ')', '<', '>']):
                print(f"⚠️ AI query contains invalid characters: '{query}', using fallback")
                return self.generate_smart_query_rule_based(topic, content_snippet)
            
            print(f"🤖 AI query generated: '{query}' for topic: '{topic}'")
            return query
            
        except Exception as e:
            print(f"⚠️ AI query generation failed: {e}")
            # Fallback to simple query
            return self._generate_fallback_query(topic)
    
    def _generate_fallback_query(self, topic):
        """
        Generate a simple fallback query when AI fails
        
        Args:
            topic (str): Content topic/title
        
        Returns:
            str: Simple search query
        """
        text = topic.lower()
        
        # Visual concept mapping - convert abstract to concrete
        visual_mappings = {
            # Technology
            'ai': 'technology',
            'artificial intelligence': 'technology',
            'machine learning': 'data analytics',
            'automation': 'technology workspace',
            'digital transformation': 'business technology',
            'cloud computing': 'data center',
            
            # Business
            'success': 'business handshake',
            'growth': 'business team',
            'strategy': 'business meeting',
            'leadership': 'business leader',
            'innovation': 'creative team',
            'productivity': 'office workspace',
            
            # Work
            'remote work': 'home office',
            'work from home': 'home office workspace',
            'collaboration': 'team meeting',
            'teamwork': 'business team',
            
            # Customer
            'customer service': 'customer support',
            'customer support': 'support team',
            'customer experience': 'happy customer',
            
            # Healthcare
            'healthcare': 'medical professional',
            'hospital': 'hospital staff',
            'medical': 'doctor',
            'patient': 'medical care',
            
            # Finance
            'finance': 'financial planning',
            'investment': 'business finance',
            'banking': 'bank professional',
        }
        
        # Check for visual mappings
        for abstract, concrete in visual_mappings.items():
            if abstract in text:
                print(f"🎯 Mapped '{abstract}' → '{concrete}'")
                return concrete
        
        # Extract industry/domain keywords
        industries = ['healthcare', 'technology', 'finance', 'education', 'retail', 
                     'manufacturing', 'marketing', 'sales', 'startup', 'business']
        
        found_industry = None
        for industry in industries:
            if industry in text:
                found_industry = industry
                break
        
        # Extract action/context keywords
        actions = ['meeting', 'working', 'using', 'team', 'office', 'professional',
                  'workspace', 'collaboration', 'planning', 'discussion']
        
        found_action = None
        for action in actions:
            if action in text:
                found_action = action
                break
        
        # Build query
        if found_industry and found_action:
            query = f"{found_industry} {found_action}"
        elif found_industry:
            query = f"{found_industry} professional"
        else:
            # Fallback to keyword extraction
            keywords = self.extract_keywords_from_topic(topic)
            query = " ".join(keywords[:2]) if keywords else topic
        
        print(f"🎯 Rule-based query: '{query}' for topic: '{topic}'")
        return query
    

    
    def generate_multiple_query_options(self, topic, content_snippet=None, count=3):
        """
        Generate multiple query options using AI and return the best ones
        
        Args:
            topic (str): Content topic
            content_snippet (str): Optional content snippet
            count (int): Number of query variations to generate
        
        Returns:
            list: List of query strings
        """
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            
            llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0.3)
            
            context = f"Topic: {topic}"
            if content_snippet:
                context += f"\nContent: {content_snippet[:500]}"
            
            prompt = f"""
            Generate {count} different search queries for finding stock photos on Pexels.
            Each query should approach the topic from a different visual angle.
            
            {context}
            
            Requirements:
            - Each query must be 2-4 words
            - Focus on different visual aspects (people, objects, settings, actions)
            - All queries must be concrete and photographable
            - Return ONLY the queries, one per line, no numbering or explanation
            
            Example for "AI in Healthcare":
            doctor using digital tablet
            hospital technology equipment
            medical professional computer screen
            """
            
            response = llm.invoke(prompt).content.strip()
            
            # Parse multiple queries
            queries = []
            for line in response.split('\n'):
                line = line.strip()
                # Remove numbering, bullets, etc.
                line = line.lstrip('0123456789.-•*) ')
                line = line.replace('"', '').replace("'", "").strip()
                
                if line and len(line.split()) >= 2:
                    queries.append(line)
            
            print(f"🤖 Generated {len(queries)} query options: {queries}")
            return queries[:count]
            
        except Exception as e:
            print(f"⚠️ Failed to generate multiple queries: {e}")
            # Fallback to single rule-based query
            return [self.generate_smart_query_rule_based(topic, content_snippet)]
    
    def get_multiple_options(self, topic, content_snippet=None, count=3):
        """
        Get multiple image options and let user/system choose the best one
        
        Args:
            topic (str): Content topic
            content_snippet (str): Optional content snippet for context
            count (int): Number of options to return
        
        Returns:
            list: List of image options with relevance scores
        """
        # Generate smart query
        smart_query = self.generate_smart_query(topic, content_snippet)
        
        # Get images
        images = self.search_images(smart_query, per_page=count)
        
        results = []
        for photo in images:
            results.append({
                "url": photo["src"]["original"],
                "url_large": photo["src"]["large"],
                "url_medium": photo["src"]["medium"],
                "photographer": photo["photographer"],
                "photographer_url": photo["photographer_url"],
                "alt": photo.get("alt", smart_query),
                "width": photo["width"],
                "height": photo["height"],
                "pexels_url": photo["url"],
                "search_query": smart_query
            })
        
        return results


if __name__ == "__main__":
    # Test the Pexels helper
    print("🧪 Testing Pexels API...")
    
    helper = PexelsHelper()
    
    # Test 1: Search for images
    print("\n📸 Test 1: Searching for 'artificial intelligence'")
    images = helper.search_images("artificial intelligence", per_page=3)
    print(f"Found {len(images)} images")
    
    # Test 2: Get best image
    print("\n📸 Test 2: Getting best image for 'business meeting'")
    image = helper.get_best_image("business meeting")
    if image:
        print(f"✅ Image URL: {image['url']}")
        print(f"📷 Photographer: {image['photographer']}")
        print(f"📐 Size: {image['width']}x{image['height']}")
    
    # Test 3: Get image for blog
    print("\n📸 Test 3: Getting image for blog about 'AI in Healthcare'")
    blog_image = helper.get_image_for_blog("AI in Healthcare")
    if blog_image:
        print(f"✅ Blog image URL: {blog_image['url']}")
    
    # Test 4: Get image for social
    print("\n📸 Test 4: Getting image for LinkedIn post")
    social_image = helper.get_image_for_social("startup growth", platform="linkedin")
    if social_image:
        print(f"✅ Social image URL: {social_image['url']}")
    
    print("\n✅ Pexels API tests complete!")

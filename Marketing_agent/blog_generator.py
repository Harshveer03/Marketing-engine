import os
import re
import json
import asyncio
from datetime import datetime
from dotenv import load_dotenv
from difflib import SequenceMatcher
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
import requests
import ollama
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

VECTOR_DB_DIR = "./vectordb"
NICHE_FILE = "./generated/niche_icp.json"
USED_TOPICS_FILE = "./generated/topics/used_blog_topics.json"
OUTPUT_FILE = "./generated/content/blogs/blogs.json"
FEEDBACK_FILE = "./generated/analytics/feedback_context.json"
SERPAPI_KEY = os.getenv("SERPAPI_KEY")


def load_feedback_context():
    if not os.path.exists(FEEDBACK_FILE):
        print("⚠️ No feedback context found. Run feedback_loop.py first.")
        return {}
    with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


class BlogGenerator:
    def __init__(self, model="models/gemini-2.5-flash", embedding_model="nomic-embed-text"):
        self.llm = ChatGoogleGenerativeAI(model=model, temperature=0.7)
        self.vectordb = None
        self.feedback = load_feedback_context()
        self.embedding_model = embedding_model
        self.niche_embedding = None
        
        # Load Engine KB for MIMIR rules
        try:
            from engine_kb_helper import EngineKBHelper
            self.engine_kb = EngineKBHelper()
            print("✅ Engine KB (MIMIR) loaded for blog generation")
        except Exception as e:
            print(f"⚠️ Engine KB not available: {e}")
            self.engine_kb = None
        
        # Load niche embedding for quality checking
        self._load_niche_embedding()
        
        # Load vector database in a thread-safe way
        self._load_vector_db_safe(embedding_model)
    
    def _load_niche_embedding(self):
        """Load and create niche embedding for quality checking"""
        try:
            niche = self.load_json(NICHE_FILE)
            if not niche:
                print("⚠️ No niche data found for quality checking")
                return
            
            # Build niche description from key fields
            niche_parts = []
            if niche.get("industry"):
                niche_parts.append(f"Industry: {niche['industry']}")
            if niche.get("value_proposition"):
                niche_parts.append(f"Value: {niche['value_proposition']}")
            if niche.get("customer_pain_points"):
                pain_points = [p.get("challenge", "") for p in niche["customer_pain_points"] if isinstance(p, dict)]
                if pain_points:
                    niche_parts.append(f"Pain Points: {', '.join(pain_points[:3])}")
            if niche.get("customer_needs"):
                needs = [n.get("need", "") for n in niche["customer_needs"] if isinstance(n, dict)]
                if needs:
                    niche_parts.append(f"Needs: {', '.join(needs[:3])}")
            
            niche_description = " | ".join(niche_parts)
            
            # Generate niche embedding
            response = ollama.embeddings(model=self.embedding_model, prompt=niche_description)
            self.niche_embedding = np.array(response['embedding'])
            print(f"✅ Niche embedding created for quality checking")
            
        except Exception as e:
            print(f"⚠️ Could not create niche embedding: {e}")
            self.niche_embedding = None
    
    def _load_vector_db_safe(self, embedding_model):
        """Load vector database in a thread-safe manner"""
        try:
            # Create a new event loop for this thread if needed
            try:
                loop = asyncio.get_event_loop()
                if loop.is_closed():
                    raise RuntimeError("Event loop is closed")
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Use a simpler approach - load without embeddings first
            import os
            if os.path.exists(VECTOR_DB_DIR):
                # Try to load with minimal embedding operations
                self.embeddings = OllamaEmbeddings(model=embedding_model)
                self.vectordb = FAISS.load_local(VECTOR_DB_DIR, self.embeddings, allow_dangerous_deserialization=True)
                print("✅ Vector database loaded successfully")
            else:
                print("⚠️ Vector database directory not found")
        except Exception as e:
            print(f"⚠️ Vector database not available: {e}")
            print("📝 Blog generation will work without PDF context")

    # ---------- Utility ----------
    def load_json(self, path):
        if not os.path.exists(path):
            # Return appropriate default based on file type
            if "niche" in path:
                return {}
            return []
        with open(path, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                # Return appropriate default based on file type
                if "niche" in path:
                    return {}
                return []

    def save_json(self, path, data):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    def append_json(self, path, new_entry):
        existing = self.load_json(path)
        if not isinstance(existing, list):
            existing = []
        existing.append(new_entry)
        self.save_json(path, existing)

    def is_similar(self, a, b, threshold=0.8):
        return SequenceMatcher(None, a.lower(), b.lower()).ratio() > threshold

    # ---------- SERPAPI News Fetch ----------
    def fetch_news(self, query):
        print(f"🔍 Fetching news for: {query}")
        url = "https://serpapi.com/search"
        params = {
            "q": query,
            "engine": "google_news",
            "num": 10,
            "api_key": SERPAPI_KEY
        }
        try:
            resp = requests.get(url, params=params, timeout=20)
            if resp.status_code != 200:
                print(f"⚠️ SerpAPI request failed with code {resp.status_code}")
                return []
            data = resp.json()
            results = []
            for art in data.get("news_results", []):
                results.append({
                    "title": art.get("title"),
                    "url": art.get("link"),
                    "snippet": art.get("snippet"),
                    "publishedAt": art.get("date", ""),
                    "source": "Google News"
                })
            return results
        except Exception as e:
            print(f"⚠️ Error fetching news: {e}")
            return []

    # ---------- Topic Generation ----------
    def generate_topic(self, niche):
        # Ensure niche is a dictionary
        if not isinstance(niche, dict):
            print(f"⚠️ Warning: niche data is not a dictionary, got {type(niche)}")
            niche = {}
            
        feedback_text = (
            self.feedback.get("blog_feedback", "") + "\n" +
            self.feedback.get("global_success_factors", "") + "\n" +
            self.feedback.get("overall_recommendation", "")
        )

        prompt = f"""
        You are a B2B SaaS marketing strategist.

        Based on this ICP/Niche JSON:
        {json.dumps(niche, indent=2)}

        Generate 1 unique, fresh blog topic (not social post) that explores industry depth — 
        long-form, analytical, and strategic — not surface-level.

        Use the following feedback insights to guide tone and topic framing:
        {feedback_text}

        Example styles: 
        - "Decoding the Future of GTM Operations in 2025"
        - "Why B2B SaaS Needs a Cultural Shift in Growth Execution"

        Return ONLY in JSON:
        {{
          "topic": "Generated topic title"
        }}
        """
        
        try:
            # Ensure we're in the right event loop context for the LLM call
            loop = None
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            response = self.llm.invoke(prompt).content
        except Exception as e:
            print(f"Error generating topic: {e}")
            # Return a fallback topic based on niche data
            industry = niche.get('industry', 'B2B SaaS')
            fallback_topics = [
                f"The Future of {industry}: Strategic Insights for 2025",
                f"Transforming {industry} Operations: A Strategic Guide",
                f"Revenue Intelligence in {industry}: Beyond Traditional Approaches",
                f"Scaling {industry} Excellence: Data-Driven Strategies",
                f"The Evolution of {industry}: Strategic Imperatives for Growth"
            ]
            import random
            return random.choice(fallback_topics)
        
        try:
            match = re.search(r"\{.*\}", response, re.DOTALL)
            if match:
                json_str = match.group()
                data = json.loads(json_str)
                return data.get("topic", response.strip())
            else:
                # Fallback if no JSON found
                return response.strip()
        except json.JSONDecodeError as e:
            print(f"⚠️ Topic generation JSON parsing error: {e}")
            print(f"Raw response: {response[:200]}...")
            # Return a cleaned version of the response
            return response.strip().replace('"', '').replace('\n', ' ')[:100]

    def generate_blog_topics_with_influence(self, trend_influence, trends_data, industry, trend_count, original_count):
        """Generate blog topics based on trend influence slider value"""
        print(f"🎯 DEBUG: Generating topics with influence: {trend_influence}")
        print(f"📈 DEBUG: Trend topics: {trend_count}, Original topics: {original_count}")
        
        topics = []
        niche = self.load_json(NICHE_FILE)
        
        # Generate trend-based topics
        for i in range(trend_count):
            if i < len(trends_data):
                trend = trends_data[i]
                print(f"📰 DEBUG: Generating trend-based topic from: {trend.get('title', 'N/A')}")
                
                # Create topic from trend
                topic = {
                    "title": trend.get('title', f"Trend Topic {i+1}"),
                    "related_news": [trend],
                    "relevance_score": trend.get('similarity_score', 75),
                    "type": "trend_follower"
                }
                topics.append(topic)
                print(f"✅ DEBUG: Added trend-based topic: {topic['title']}")
        
        # Generate original topics (trend setters)
        for i in range(original_count):
            print(f"💡 DEBUG: Generating original topic {i+1}/{original_count}")
            
            # Use a hybrid approach: reference trends but create unique angles
            prompt = f"""
            You are a B2B SaaS marketing strategist creating ORIGINAL, trend-setting content.
            
            Industry: {industry}
            Niche Context: {json.dumps(niche, indent=2) if niche else 'N/A'}
            
            {"Recent trends for context (create unique angles, don't copy):" if trends_data else ""}
            {json.dumps([t.get('title', '') for t in trends_data[:3]], indent=2) if trends_data else ''}
            
            Generate 1 ORIGINAL blog topic that:
            - Creates a NEW perspective or trend in {industry}
            - Goes beyond current discussions
            - Provides strategic, forward-thinking insights
            - Is analytical and thought-provoking
            
            Examples of trend-setting topics:
            - "Why {industry} Needs to Rethink [Common Practice]"
            - "The Hidden Cost of [Industry Standard] in {industry}"
            - "Beyond [Current Trend]: What's Next for {industry}"
            
            Return ONLY in JSON:
            {{
              "topic": "Your original topic title",
              "angle": "Brief description of the unique angle"
            }}
            """
            
            try:
                # Ensure event loop
                loop = None
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                response = self.llm.invoke(prompt).content
                
                # Parse response
                match = re.search(r"\{.*\}", response, re.DOTALL)
                if match:
                    json_str = match.group()
                    data = json.loads(json_str)
                    topic_title = data.get("topic", f"Original Topic {i+1}")
                else:
                    topic_title = response.strip()[:100]
                
                # Create original topic
                topic = {
                    "title": topic_title,
                    "related_news": trends_data[:2] if trends_data else [],  # Reference some trends for context
                    "relevance_score": 85,  # Higher score for original content
                    "type": "trend_setter"
                }
                topics.append(topic)
                print(f"✅ DEBUG: Added original topic: {topic['title']}")
                
            except Exception as e:
                print(f"❌ DEBUG: Error generating original topic: {e}")
                # Fallback original topic
                fallback_topics = [
                    f"Rethinking {industry}: A Strategic Imperative for 2025",
                    f"The Future of {industry}: Beyond Current Trends",
                    f"Why {industry} Leaders Are Missing the Bigger Picture",
                    f"The Hidden Opportunity in {industry} Transformation",
                    f"Breaking the {industry} Status Quo: A New Approach"
                ]
                topic = {
                    "title": fallback_topics[i % len(fallback_topics)],
                    "related_news": trends_data[:2] if trends_data else [],
                    "relevance_score": 80,
                    "type": "trend_setter"
                }
                topics.append(topic)
                print(f"⚠️ DEBUG: Used fallback original topic: {topic['title']}")
        
        print(f"✅ DEBUG: Generated total {len(topics)} topics")
        return topics

    # ---------- PDF Context ----------
    def build_pdf_context(self, query_text, niche=None):
        # Build context from niche data first
        context_parts = []
        
        if niche:
            context_parts.append(self._build_context_from_niche(niche))
        
        # Try vector search as additional context if available
        if self.vectordb:
            try:
                docs = self.vectordb.similarity_search(query_text, k=5)
                vector_context = "\n".join([doc.page_content for doc in docs])
                context_parts.append(f"Additional Context:\n{vector_context}")
            except Exception as e:
                print(f"Vector search failed, using niche context: {e}")
        
        return "\n".join(context_parts) if context_parts else "Using niche data for context"
    
    def _build_context_from_niche(self, niche):
        """Build rich context from niche data"""
        context_parts = []
        
        if niche.get("industry"):
            context_parts.append(f"Industry: {niche['industry']}")
        
        if niche.get("value_proposition"):
            context_parts.append(f"Value Proposition: {niche['value_proposition']}")
        
        if niche.get("customer_pain_points"):
            pain_points = []
            for pain in niche["customer_pain_points"]:
                if isinstance(pain, dict) and pain.get("challenge"):
                    pain_points.append(pain["challenge"])
            if pain_points:
                context_parts.append(f"Customer Pain Points: {'; '.join(pain_points)}")
        
        if niche.get("customer_needs"):
            needs = []
            for need in niche["customer_needs"]:
                if isinstance(need, dict) and need.get("need"):
                    needs.append(need["need"])
            if needs:
                context_parts.append(f"Customer Needs: {'; '.join(needs)}")
        
        if niche.get("target_audience"):
            context_parts.append(f"Target Audience: {', '.join(niche['target_audience'])}")
        
        return "\n".join(context_parts)

    def _clean_json_string(self, json_str):
        """Clean JSON string to handle control characters and formatting issues"""
        import json as json_module
        
        # First, try to find and extract just the JSON content
        try:
            # Remove any leading/trailing whitespace and non-JSON content
            json_str = json_str.strip()
            
            # Find the actual JSON object boundaries
            start = json_str.find('{')
            end = json_str.rfind('}') + 1
            
            if start != -1 and end > start:
                json_str = json_str[start:end]
            
            # Replace problematic characters that might break JSON parsing
            # Handle unescaped newlines in strings
            json_str = re.sub(r'(?<!\\)\n', '\\n', json_str)
            json_str = re.sub(r'(?<!\\)\r', '\\r', json_str)
            json_str = re.sub(r'(?<!\\)\t', '\\t', json_str)
            
            # Handle unescaped quotes (this is tricky, so we'll be conservative)
            # Only replace quotes that are clearly not part of JSON structure
            json_str = re.sub(r'(?<!\\)"(?=\w)', '\\"', json_str)
            
            return json_str
            
        except Exception as e:
            print(f"Error cleaning JSON string: {e}")
            return json_str
    
    def _extract_content_manually(self, response, topic):
        """Manually extract content when JSON parsing fails"""
        try:
            print(f"🔧 Attempting manual content extraction for topic: {topic}")
            
            # Try to extract title
            title_match = re.search(r'"title":\s*"([^"]*)"', response)
            title = title_match.group(1) if title_match else topic
            
            # Try to extract blog content - improved regex to handle nested JSON
            # Look for the blog content within the JSON structure
            blog_match = re.search(r'"blog":\s*"(.*?)"\s*(?:,\s*"|\s*})', response, re.DOTALL)
            if blog_match:
                blog_content = blog_match.group(1)
                # Clean up escaped characters
                blog_content = blog_content.replace('\\"', '"').replace('\\n', '\n').replace('\\t', '\t').replace('\\r', '\r')
                print(f"✅ Successfully extracted blog content ({len(blog_content)} characters)")
            else:
                # Alternative approach: try to find content between blog field markers
                blog_start = response.find('"blog": "')
                if blog_start != -1:
                    blog_start += len('"blog": "')
                    # Find the end of the blog content (look for closing quote followed by comma or brace)
                    blog_end = response.find('"}', blog_start)
                    if blog_end == -1:
                        blog_end = response.find('",', blog_start)
                    if blog_end != -1:
                        blog_content = response[blog_start:blog_end]
                        blog_content = blog_content.replace('\\"', '"').replace('\\n', '\n').replace('\\t', '\t')
                        print(f"✅ Extracted blog content using alternative method ({len(blog_content)} characters)")
                    else:
                        # Last resort: use the whole response but clean it up
                        blog_content = response.strip()
                        # Remove JSON structure markers
                        blog_content = re.sub(r'^\{.*?"blog":\s*"', '', blog_content, flags=re.DOTALL)
                        blog_content = re.sub(r'"\s*\}.*$', '', blog_content, flags=re.DOTALL)
                        blog_content = blog_content.replace('\\"', '"').replace('\\n', '\n')
                        print(f"⚠️ Used fallback extraction method ({len(blog_content)} characters)")
                else:
                    # If no blog field found, use the whole response
                    blog_content = response.strip()
                    print(f"⚠️ No blog field found, using entire response ({len(blog_content)} characters)")
            
            # Try to extract outline
            outline_match = re.search(r'"outline":\s*\[(.*?)\]', response)
            if outline_match:
                outline_str = outline_match.group(1)
                # Clean up the outline items
                outline_items = []
                for item in outline_str.split(','):
                    clean_item = item.strip().strip('"').strip("'")
                    if clean_item:
                        outline_items.append(clean_item)
                outline = outline_items if outline_items else ["Introduction", "Analysis", "Solutions", "Conclusion"]
            else:
                outline = ["Introduction", "Analysis", "Solutions", "Conclusion"]
            
            print(f"📝 Manual extraction complete - Title: {title}, Outline items: {len(outline)}")
            
            return {
                "title": title,
                "outline": outline,
                "blog": blog_content
            }
            
        except Exception as e:
            print(f"❌ Error in manual content extraction: {e}")
            return {
                "title": topic,
                "outline": ["Introduction", "Analysis", "Solutions", "Conclusion"],
                "blog": "Content extraction failed. Please regenerate this blog post."
            }

    # ---------- Blog Generation ----------
    def generate_blog_with_industry(self, topic, news_items, niche, pdf_context, industry=None, tone="professional", audience="CXOs"):
        """Generate blog with industry-specific context"""
        # Ensure niche is a dictionary
        if not isinstance(niche, dict):
            print(f"⚠️ Warning: niche data is not a dictionary, got {type(niche)}")
            niche = {}
        
        feedback_text = (
            self.feedback.get("blog_feedback", "") + "\n" +
            self.feedback.get("global_success_factors", "") + "\n" +
            self.feedback.get("overall_recommendation", "")
        )

        # Target industry is what user selects from frontend (who they want to target)
        target_industry = industry or "B2B SaaS"
        # Original industry is the user's own business sector from niche data
        original_industry = niche.get("industry", "B2B SaaS")
        
        # Safely extract pain points and needs
        pain_points = []
        if isinstance(niche.get('customer_pain_points'), list):
            pain_points = [p.get('challenge', '') for p in niche.get('customer_pain_points', []) if isinstance(p, dict)]
        
        customer_needs = []
        if isinstance(niche.get('customer_needs'), list):
            customer_needs = [n.get('need', '') for n in niche.get('customer_needs', []) if isinstance(n, dict)]

        print(f"📝 Blog: Generating content for topic: '{topic}' in {target_industry} industry")

        # Query MIMIR rules for blog generation
        mimir_rules = ""
        if self.engine_kb and self.engine_kb.vectordb:
            try:
                print(f"\n🧠 Fetching MIMIR rules for Blog generation...")
                mimir_rules = self.engine_kb.get_blog_rules(
                    topic=topic,
                    audience=audience,
                    tone=tone
                )
                print(f"✅ MIMIR rules loaded: {len(mimir_rules)} chars\n")
            except Exception as e:
                print(f"⚠️ Could not load MIMIR rules: {e}")
                mimir_rules = ""

        # Build prompt with MIMIR integration
        mimir_section = ""
        if mimir_rules:
            mimir_section = f"""

{'='*60}
MIMIR CONTENT GENERATION RULES (FOLLOW STRICTLY):
{'='*60}
{mimir_rules}

Apply MIMIR framework for blog generation:
- Intent & Grounding (Part 1): Clear message intent and audience focus
- Phrasing Foundations (Part 2): Strong sentence construction
- Structure Architecture (Part 3): Logical content hierarchy
- Tone Decision (Part 4): Maintain {tone} tone throughout
- Tailoring Principles (Part 7): Adapt to {audience} in {target_industry}
- Narrative Physics (Part 12): Story logic, rhythm, and pacing
- Integrity & Grounding (Part 13): Factual accuracy, no hallucination
- Logic-Emotion Balance (Part 15): Persuasive yet authentic
{'='*60}
"""

        prompt = f"""
        You are an expert B2B SaaS content strategist.
        {mimir_section}

        Write a comprehensive blog on the topic: "{topic}"

        Context:
        - Target Industry (Blog Audience): {target_industry}
        - Your Business Industry: {original_industry}
        - Key Pain Points: {pain_points}
        - Customer Needs: {customer_needs}
        - Target Audience: {audience}
        - Desired Tone: {tone}
        - Relevant News Articles: {json.dumps(news_items, indent=2)}
        - Reference Material: {pdf_context[:2000] if pdf_context else "No additional context available"}

        Use the following performance feedback to guide writing tone, structure, and topic positioning:
        {feedback_text}

        Blog Requirements:
        1. Write a well-structured, long-form blog (700–1000 words) specifically for {target_industry} professionals.
        2. Write from the perspective of a {original_industry} expert addressing {target_industry} challenges.
        3. Include clear sections: Introduction, Core Analysis, Solutions/Insights, and Conclusion.
        4. Tone: {tone.capitalize()}, forward-thinking, and authoritative for {audience} in {target_industry}.
        5. Include subtle references to {target_industry} trends and challenges.
        6. Address {target_industry}-specific pain points and opportunities from a {original_industry} solution perspective.
        7. Do NOT add markdown or emojis.

        Output in JSON:
        {{
          "title": "{topic}",
          "outline": ["Introduction", "Analysis", "Solutions", "Conclusion"],
          "blog": "Full text here..."
        }}
        """
        
        try:
            # Ensure we're in the right event loop context for the LLM call
            loop = None
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            response = self.llm.invoke(prompt).content
            print(f"✅ Blog content generated successfully for {target_industry}")
        except Exception as e:
            print(f"Error generating blog content: {e}")
            # Return a fallback blog with rich content based on niche data
            return self._generate_fallback_blog_with_industry(topic, niche, news_items, target_industry, original_industry, tone, audience)
        
        # Try to extract and parse JSON with better error handling
        try:
            match = re.search(r"\{.*\}", response, re.DOTALL)
            if match:
                json_str = match.group()
                # Clean the JSON string to handle control characters
                json_str = self._clean_json_string(json_str)
                data = json.loads(json_str)
            else:
                # Fallback if no JSON found
                data = {
                    "title": topic,
                    "outline": ["Introduction", "Analysis", "Solutions", "Conclusion"],
                    "blog": response.strip()
                }
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON parsing error: {e}")
            print(f"Raw response: {response[:500]}...")
            # Try to extract content manually as fallback
            data = self._extract_content_manually(response, topic)
        
        return data
    
    def _generate_fallback_blog_with_industry(self, topic, niche, news_items, target_industry, original_industry, tone, audience):
        """Generate a fallback blog with both target and original industry context when AI generation fails"""
        value_prop = niche.get('value_proposition', 'innovative solutions')
        
        # Extract pain points
        pain_points = []
        if isinstance(niche.get('customer_pain_points'), list):
            pain_points = [p.get('challenge', '') for p in niche.get('customer_pain_points', []) if isinstance(p, dict)]
        
        # Generate a structured blog with both industry contexts
        blog_content = f"""Introduction

The {target_industry} landscape is rapidly evolving, presenting both unprecedented opportunities and complex challenges for organizations seeking sustainable growth. As market dynamics shift and customer expectations continue to rise, traditional approaches are proving insufficient to meet the demands of modern {target_industry} environments.

As {original_industry} experts, we understand the unique challenges facing today's {audience.lower()} in {target_industry}. The intersection of {original_industry} solutions and {target_industry} needs creates a critical inflection point where {topic.lower()} is becoming not just an advantage, but a necessity for competitive survival and growth.

Core Analysis: The Current Challenge in {target_industry}

{target_industry} organizations today face several critical challenges that require immediate attention and strategic response. {pain_points[0] if pain_points else f'Organizations in {target_industry} struggle with operational efficiency and market positioning.'} This fundamental issue impacts not only immediate performance but also long-term strategic positioning in an increasingly competitive {target_industry} marketplace.

{pain_points[1] if len(pain_points) > 1 else f'Additionally, the complexity of modern {target_industry} operations requires sophisticated approaches to data management and customer engagement.'} These challenges compound to create significant barriers to growth and operational excellence in the {target_industry} sector.

Strategic Solutions and Insights for {target_industry}

Drawing from our expertise in {original_industry}, the path forward for {target_industry} organizations requires a systematic approach that addresses these challenges through {value_prop}. {audience} must focus on three key areas:

First, implementing data-driven decision-making processes that transform raw information into actionable insights specific to {target_industry} operations. This involves not just collecting data, but developing the analytical capabilities to extract meaningful patterns and trends that inform strategic decisions in the {target_industry} context.

Second, developing scalable operational frameworks that can adapt to changing {target_industry} conditions while maintaining consistency in service delivery and customer experience. This requires both technological infrastructure and organizational capabilities that support rapid scaling without compromising quality in {target_industry} operations.

Third, fostering a culture of continuous improvement and innovation that enables {target_industry} organizations to stay ahead of market trends and customer needs. This involves investing in team development, process optimization, and strategic partnerships that enhance overall {target_industry} capabilities.

Conclusion: The Path Forward for {target_industry}

The future belongs to {target_industry} organizations that can successfully navigate the complexities of the modern business environment while delivering exceptional value to their customers. By leveraging {original_industry} expertise and focusing on {value_prop}, {audience.lower()} can position their organizations for sustained success and market leadership in {target_industry}.

The time for incremental change has passed. {target_industry} organizations must embrace transformative approaches that address root causes rather than symptoms, building capabilities that will serve them well in an uncertain and rapidly changing {target_industry} future."""

        return {
            "title": topic,
            "outline": ["Introduction", f"Core Analysis: The Current Challenge in {target_industry}", f"Strategic Solutions and Insights for {target_industry}", f"Conclusion: The Path Forward for {target_industry}"],
            "blog": blog_content
        }
    
    # ---------- Quality Checker ----------
    def calculate_quality_score(self, topic, content, trends):
        """Calculate quality score by comparing topic, content, and trends against niche embedding"""
        if self.niche_embedding is None:
            print("⚠️ Niche embedding not available, skipping quality check")
            return 0
        
        try:
            # Generate embeddings for topic, content, and trends
            print("🔍 Generating embeddings for quality check...")
            
            # Topic embedding
            topic_response = ollama.embeddings(model=self.embedding_model, prompt=topic)
            topic_embedding = np.array(topic_response['embedding'])
            
            # Content embedding (use first 1000 chars to avoid token limits)
            content_sample = content[:1000] if len(content) > 1000 else content
            content_response = ollama.embeddings(model=self.embedding_model, prompt=content_sample)
            content_embedding = np.array(content_response['embedding'])
            
            # Trends embedding (combine trend titles)
            if isinstance(trends, list) and trends:
                trends_text = " | ".join([t.get('title', '') for t in trends[:5] if isinstance(t, dict)])
            else:
                trends_text = "No trends available"
            trends_response = ollama.embeddings(model=self.embedding_model, prompt=trends_text)
            trends_embedding = np.array(trends_response['embedding'])
            
            # Calculate cosine similarities
            topic_similarity = cosine_similarity([topic_embedding], [self.niche_embedding])[0][0]
            content_similarity = cosine_similarity([content_embedding], [self.niche_embedding])[0][0]
            trends_similarity = cosine_similarity([trends_embedding], [self.niche_embedding])[0][0]
            
            # Weighted quality score: Content 50%, Topic 30%, Trends 20%
            quality_score = (content_similarity * 0.5 + topic_similarity * 0.3 + trends_similarity * 0.2) * 100
            
            print(f"📊 Quality Scores - Topic: {topic_similarity:.2f}, Content: {content_similarity:.2f}, Trends: {trends_similarity:.2f}")
            print(f"✅ Overall Quality Score: {quality_score:.1f}%")
            
            return round(quality_score, 2)
            
        except Exception as e:
            print(f"⚠️ Error calculating quality score: {e}")
            return 0


    # ---------- Execution Flow ----------
    def run(self, mode="manual"):
        try:
            niche = self.load_json(NICHE_FILE)
            used_topics = self.load_json(USED_TOPICS_FILE)

            if mode == "manual":
                topic = input("Enter your blog topic: ").strip()
                news_items = self.fetch_news(topic)
            else:
                print("🤖 Generating new blog topic from niche...")
                topic = self.generate_topic(niche)

                while any(self.is_similar(topic, t.get("title", "")) for t in used_topics if isinstance(t, dict)):
                    print("⚠️ Duplicate topic detected, regenerating...")
                    topic = self.generate_topic(niche)

                news_items = self.fetch_news(topic)

            pdf_context = self.build_pdf_context(topic, niche)
            # Use industry from niche data, with defaults for tone and audience
            industry = niche.get("industry", "B2B SaaS")
            blog_data = self.generate_blog_with_industry(topic, news_items, niche, pdf_context, industry, "professional", "CXOs")

            # Calculate quality score
            quality_score = self.calculate_quality_score(
                topic,
                blog_data.get("blog", ""),
                news_items
            )

            # ✅ Append new blog to single JSON file
            blog_entry = {
                "title": blog_data.get("title", topic),
                "outline": blog_data.get("outline", []),
                "blog": blog_data.get("blog", ""),
                "news": news_items,
                "quality_score": quality_score,
                "timestamp": datetime.utcnow().isoformat()
            }
            self.append_json(OUTPUT_FILE, blog_entry)
            
            # Save used topic AFTER successful blog generation (only for automatic mode)
            if mode != "manual":
                self.append_json(USED_TOPICS_FILE, {"title": topic, "generated_on": datetime.utcnow().isoformat()})
                print(f"📝 Saved used topic: '{topic}' to {USED_TOPICS_FILE}")

            print(f"\n✅ Blog appended to: {OUTPUT_FILE}")
            print(f"📝 Title: {blog_entry['title']}")
            return blog_entry
            
        except Exception as e:
            print(f"❌ Error in blog generation: {e}")
            import traceback
            traceback.print_exc()
            return {"title": "Error generating blog", "error": str(e)}


if __name__ == "__main__":
    print("\n--- Blog Generator ---")
    mode = input("Choose mode (manual / automatic): ").strip().lower()
    if mode not in ["manual", "automatic"]:
        mode = "manual"
    gen = BlogGenerator()
    gen.run(mode)

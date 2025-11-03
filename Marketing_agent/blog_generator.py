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
        
        # Load vector database in a thread-safe way
        self._load_vector_db_safe(embedding_model)
    
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
                self.embeddings = OllamaEmbeddingsEmbeddings(model=embedding_model)
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

    # ---------- Blog Generation ----------
    def generate_blog(self, topic, news_items, niche, pdf_context):
        # Ensure niche is a dictionary
        if not isinstance(niche, dict):
            print(f"⚠️ Warning: niche data is not a dictionary, got {type(niche)}")
            niche = {}
        
        feedback_text = (
            self.feedback.get("blog_feedback", "") + "\n" +
            self.feedback.get("global_success_factors", "") + "\n" +
            self.feedback.get("overall_recommendation", "")
        )

        # Safely extract pain points and needs
        pain_points = []
        if isinstance(niche.get('customer_pain_points'), list):
            pain_points = [p.get('challenge', '') for p in niche.get('customer_pain_points', []) if isinstance(p, dict)]
        
        customer_needs = []
        if isinstance(niche.get('customer_needs'), list):
            customer_needs = [n.get('need', '') for n in niche.get('customer_needs', []) if isinstance(n, dict)]

        prompt = f"""
        You are an expert B2B SaaS content strategist.

        Write a comprehensive blog on the topic: "{topic}"

        Context:
        - Industry: {niche.get("industry", "B2B SaaS")}
        - Key Pain Points: {pain_points}
        - Customer Needs: {customer_needs}
        - Relevant News Articles: {json.dumps(news_items, indent=2)}
        - Reference Material: {pdf_context[:2000] if pdf_context else "No additional context available"}

        Use the following performance feedback to guide writing tone, structure, and topic positioning:
        {feedback_text}

        Blog Requirements:
        1. Write a well-structured, long-form blog (700–1000 words).
        2. Include clear sections: Introduction, Core Analysis, Solutions/Insights, and Conclusion.
        3. Tone: Analytical, forward-thinking, and authoritative.
        4. Include subtle references to recent industry shifts.
        5. Do NOT add markdown or emojis.

        Output in JSON:
        {{
          "title": "{topic}",
          "outline": ["Intro", "Main Insight 1", "Main Insight 2", "Conclusion"],
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
        except Exception as e:
            print(f"Error generating blog content: {e}")
            # Return a fallback blog with rich content based on niche data
            return self._generate_fallback_blog(topic, niche, news_items)
        
        # Try to extract and parse JSON with better error handling
        try:
            match = re.search(r"\{.*\}", response, re.DOTALL)
            if match:
                json_str = match.group()
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
            # Fallback with safe data
            data = {
                "title": topic,
                "outline": ["Introduction", "Analysis", "Solutions", "Conclusion"],
                "blog": response.strip()
            }
        
        return data
    
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

        # Use provided industry or fall back to niche industry
        target_industry = industry or niche.get("industry", "B2B SaaS")
        
        # Safely extract pain points and needs
        pain_points = []
        if isinstance(niche.get('customer_pain_points'), list):
            pain_points = [p.get('challenge', '') for p in niche.get('customer_pain_points', []) if isinstance(p, dict)]
        
        customer_needs = []
        if isinstance(niche.get('customer_needs'), list):
            customer_needs = [n.get('need', '') for n in niche.get('customer_needs', []) if isinstance(n, dict)]

        print(f"📝 Blog: Generating content for topic: '{topic}' in {target_industry} industry")

        prompt = f"""
        You are an expert B2B SaaS content strategist.

        Write a comprehensive blog on the topic: "{topic}"

        Context:
        - Target Industry: {target_industry}
        - Original Industry Context: {niche.get("industry", "B2B SaaS")}
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
        2. Include clear sections: Introduction, Core Analysis, Solutions/Insights, and Conclusion.
        3. Tone: {tone.capitalize()}, forward-thinking, and authoritative for {audience} in {target_industry}.
        4. Include subtle references to {target_industry} trends and challenges.
        5. Address {target_industry}-specific pain points and opportunities.
        6. Do NOT add markdown or emojis.

        Output in JSON:
        {{
          "title": "{topic}",
          "outline": ["Intro", "Main Insight 1", "Main Insight 2", "Conclusion"],
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
            return self._generate_fallback_blog_with_industry(topic, niche, news_items, target_industry, tone, audience)
        
        # Try to extract and parse JSON with better error handling
        try:
            match = re.search(r"\{.*\}", response, re.DOTALL)
            if match:
                json_str = match.group()
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
            # Fallback with safe data
            data = {
                "title": topic,
                "outline": ["Introduction", "Analysis", "Solutions", "Conclusion"],
                "blog": response.strip()
            }
        
        return data
    
    def _generate_fallback_blog_with_industry(self, topic, niche, news_items, industry, tone, audience):
        """Generate a fallback blog with industry context when AI generation fails"""
        value_prop = niche.get('value_proposition', 'innovative solutions')
        
        # Extract pain points
        pain_points = []
        if isinstance(niche.get('customer_pain_points'), list):
            pain_points = [p.get('challenge', '') for p in niche.get('customer_pain_points', []) if isinstance(p, dict)]
        
        # Generate a structured blog with industry context
        blog_content = f"""Introduction

The {industry} landscape is rapidly evolving, presenting both unprecedented opportunities and complex challenges for organizations seeking sustainable growth. As market dynamics shift and customer expectations continue to rise, traditional approaches are proving insufficient to meet the demands of modern {industry} environments.

Today's {audience.lower()} in {industry} face a critical inflection point where {topic.lower()} is becoming not just an advantage, but a necessity for competitive survival and growth.

Core Analysis: The Current Challenge in {industry}

{industry} organizations today face several critical challenges that require immediate attention and strategic response. {pain_points[0] if pain_points else f'Organizations in {industry} struggle with operational efficiency and market positioning.'} This fundamental issue impacts not only immediate performance but also long-term strategic positioning in an increasingly competitive {industry} marketplace.

{pain_points[1] if len(pain_points) > 1 else f'Additionally, the complexity of modern {industry} operations requires sophisticated approaches to data management and customer engagement.'} These challenges compound to create significant barriers to growth and operational excellence in the {industry} sector.

Strategic Solutions and Insights for {industry}

The path forward for {industry} organizations requires a systematic approach that addresses these challenges through {value_prop}. {audience} must focus on three key areas:

First, implementing data-driven decision-making processes that transform raw information into actionable insights specific to {industry} operations. This involves not just collecting data, but developing the analytical capabilities to extract meaningful patterns and trends that inform strategic decisions in the {industry} context.

Second, developing scalable operational frameworks that can adapt to changing {industry} conditions while maintaining consistency in service delivery and customer experience. This requires both technological infrastructure and organizational capabilities that support rapid scaling without compromising quality in {industry} operations.

Third, fostering a culture of continuous improvement and innovation that enables {industry} organizations to stay ahead of market trends and customer needs. This involves investing in team development, process optimization, and strategic partnerships that enhance overall {industry} capabilities.

Conclusion: The Path Forward for {industry}

The future belongs to {industry} organizations that can successfully navigate the complexities of the modern business environment while delivering exceptional value to their customers. By focusing on {value_prop} and addressing the fundamental challenges outlined above, {audience.lower()} can position their organizations for sustained success and market leadership in {industry}.

The time for incremental change has passed. {industry} organizations must embrace transformative approaches that address root causes rather than symptoms, building capabilities that will serve them well in an uncertain and rapidly changing {industry} future."""

        return {
            "title": topic,
            "outline": ["Introduction", f"Core Analysis: The Current Challenge in {industry}", f"Strategic Solutions and Insights for {industry}", f"Conclusion: The Path Forward for {industry}"],
            "blog": blog_content
        }
    
    def _generate_fallback_blog(self, topic, niche, news_items):
        """Generate a fallback blog when AI generation fails"""
        industry = niche.get('industry', 'B2B SaaS')
        value_prop = niche.get('value_proposition', 'innovative solutions')
        
        # Extract pain points
        pain_points = []
        if isinstance(niche.get('customer_pain_points'), list):
            pain_points = [p.get('challenge', '') for p in niche.get('customer_pain_points', []) if isinstance(p, dict)]
        
        # Generate a structured blog
        blog_content = f"""Introduction

The {industry} landscape is rapidly evolving, presenting both unprecedented opportunities and complex challenges for organizations seeking sustainable growth. As market dynamics shift and customer expectations continue to rise, traditional approaches are proving insufficient to meet the demands of modern business environments.

Core Analysis: The Current Challenge

Today's {industry} organizations face several critical challenges that require immediate attention and strategic response. {pain_points[0] if pain_points else 'Organizations struggle with operational efficiency and market positioning.'} This fundamental issue impacts not only immediate performance but also long-term strategic positioning in an increasingly competitive marketplace.

{pain_points[1] if len(pain_points) > 1 else 'Additionally, the complexity of modern business operations requires sophisticated approaches to data management and customer engagement.'} These challenges compound to create significant barriers to growth and operational excellence.

Solutions and Strategic Insights

The path forward requires a systematic approach that addresses these challenges through {value_prop}. Organizations must focus on three key areas:

First, implementing data-driven decision-making processes that transform raw information into actionable insights. This involves not just collecting data, but developing the analytical capabilities to extract meaningful patterns and trends that inform strategic decisions.

Second, developing scalable operational frameworks that can adapt to changing market conditions while maintaining consistency in service delivery and customer experience. This requires both technological infrastructure and organizational capabilities that support rapid scaling without compromising quality.

Third, fostering a culture of continuous improvement and innovation that enables organizations to stay ahead of market trends and customer needs. This involves investing in team development, process optimization, and strategic partnerships that enhance overall capabilities.

Conclusion: The Path Forward

The future belongs to organizations that can successfully navigate the complexities of the modern {industry} environment while delivering exceptional value to their customers. By focusing on {value_prop} and addressing the fundamental challenges outlined above, organizations can position themselves for sustained success and market leadership.

The time for incremental change has passed. Organizations must embrace transformative approaches that address root causes rather than symptoms, building capabilities that will serve them well in an uncertain and rapidly changing future."""

        return {
            "title": topic,
            "outline": ["Introduction", "Core Analysis: The Current Challenge", "Solutions and Strategic Insights", "Conclusion: The Path Forward"],
            "blog": blog_content
        }

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
                self.append_json(USED_TOPICS_FILE, {"title": topic, "generated_on": datetime.utcnow().isoformat()})

            pdf_context = self.build_pdf_context(topic, niche)
            blog_data = self.generate_blog(topic, news_items, niche, pdf_context)

            # ✅ Append new blog to single JSON file
            blog_entry = {
                "title": blog_data.get("title", topic),
                "outline": blog_data.get("outline", []),
                "blog": blog_data.get("blog", ""),
                "news": news_items,
                "timestamp": datetime.utcnow().isoformat()
            }
            self.append_json(OUTPUT_FILE, blog_entry)

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

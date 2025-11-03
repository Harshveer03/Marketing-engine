import os
import re
import json
import asyncio
from dotenv import load_dotenv
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings

load_dotenv()

NEWS_FILE = "./generated/news/filtered_news.json"
TOPICS_FILE = "./generated/topics/topics.json"
NICHE_FILE = "./generated/niche_icp.json"
OUTPUT_DIR = "./generated/content/social"
VECTOR_DB_DIR = "./vectordb"

def load_feedback_context():
    FEEDBACK_FILE = "./generated/analytics/feedback_context.json"
    if not os.path.exists(FEEDBACK_FILE):
        print("⚠️ No feedback context found. Run feedback_loop.py first.")
        return {}
    with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


class ContentPipeline:
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
                self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
                self.vectordb = FAISS.load_local(VECTOR_DB_DIR, self.embeddings, allow_dangerous_deserialization=True)
                print("✅ Vector database loaded successfully")
            else:
                print("⚠️ Vector database directory not found")
        except Exception as e:
            print(f"⚠️ Vector database not available: {e}")
            print("📝 Content generation will work without PDF context")

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
        os.makedirs(os.path.dirname(path), exist_ok=True)
        existing = self.load_json(path)
        if not isinstance(existing, list):
            existing = []
        existing.append(new_entry)
        self.save_json(path, existing)

    def clean_response(self, response):
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", response, re.S)
            return json.loads(match.group()) if match else {}

    # ---------- Topic Generation ----------
    def generate_topics(self):
        news_list = self.load_json(NEWS_FILE)
        if not news_list:
            raise FileNotFoundError("⚠ No news data found. Run trend_fetcher first.")
        
        used_list = self.load_json(TOPICS_FILE)
        used_text = "\n".join([f"- {t['title']}" for t in used_list]) if used_list else "None"

        feedback_summary = (
            self.feedback.get("overall_recommendation", "") + "\n" +
            self.feedback.get("global_success_factors", "")
        )
        
        prompt = f"""
        You are an AI assistant that generates strategic social media content topics.
        Input:
        1. A list of recent news/trends.
        2. A list of previously used topics.
        Your job:
        1. Analyze the news/trends deeply to detect patterns, overlaps, and recurring themes. 
            - Look for strategic signals: shifts in customer behavior, regulatory changes, competitive plays, tech adoption, or market risks.
        2. Merge related items into a single concise theme if possible (avoid redundancy).
        3. Generate exactly 3 NEW, distinct, and high-relevance topic options that are NOT in the previously used list.
        4. Each topic MUST:
            - Be short (max 12 words).
            - Be phrased as a clear, actionable content hook.
            - Be directly relevant to CXO-level priorities (growth, risk, innovation, market shifts).
            - Highlight opportunities, threats, or decisions executives care about.
            - Avoid vague wording or generic filler (e.g., no "future of", "latest insights").
        5. Do not repeat the same words or phrasing across the 3 topics.
        
        Use the following performance feedback to guide topic selection and phrasing:
        {feedback_summary}
        
        News Items:
        {news_list}
        
        Previously Used Topics:
        {used_text}
        
        Output ONLY in valid JSON as:
        {{
            "topics": [
                {{"title": "Topic Title 1"}},
                {{"title": "Topic Title 2"}},
                {{"title": "Topic Title 3"}}
            ]
        }}
        """
        response = self.llm.invoke(prompt).content.strip()
        data = self.clean_response(response)
        topics = data.get("topics", [])

        for t in topics:
            t["related_news"] = [
                n for n in news_list if any(
                    word.lower() in ((n.get("title") or "") + " " + (n.get("description") or "")).lower()
                    for word in t["title"].split()
                )
            ][:5]

        return topics

    # ---------- Context Builders ----------
    def get_context(self, topic):
        niche = self.load_json(NICHE_FILE)
        
        # Create rich context from niche data instead of vector search
        pdf_context = self._build_context_from_niche(niche)
        
        # Try vector search as fallback if available
        if self.vectordb:
            try:
                query_text = topic["title"]
                pdf_docs = self.vectordb.similarity_search(query_text, k=3)
                vector_context = "\n".join([doc.page_content for doc in pdf_docs])
                pdf_context = f"{pdf_context}\n\nAdditional Context:\n{vector_context}"
            except Exception as e:
                print(f"Vector search failed, using niche context: {e}")
        
        return niche, pdf_context
    
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

    def _format_pain_points(self, niche):
        pain_points_list = []
        for item in niche.get('customer_pain_points', []):
            challenge = item.get('challenge', '')
            why_details = item.get('why', [])
            explanations = []
            for w in why_details:
                cause = w.get('cause', '')
                explanation = w.get('explanation', '')
                indicators = "; ".join(w.get('indicators', []))
                explanations.append(f"Cause: {cause}, Explanation: {explanation}, Indicators: {indicators}")
            full_detail = f"{challenge} [{' | '.join(explanations)}]"
            pain_points_list.append(full_detail)
        return "\n- ".join(pain_points_list)

    def _format_needs(self, niche):
        return ', '.join([item['need'] for item in niche.get('customer_needs', [])])

    # ---------- Content Generation ----------
    def generate_linkedin(self, topic, related_news, niche, audience, tone, pdf_context, industry=None):
        try:
            feedback_text = self.feedback.get("linkedin_feedback", "")
            pain_points = self._format_pain_points(niche)
            needs = self._format_needs(niche)
            
            # Use provided industry or fall back to niche industry
            target_industry = industry or niche.get("industry", "Technology")
            
            # Debug: Ensure we're using the correct topic
            topic_title = topic['title'] if isinstance(topic, dict) else str(topic)
            print(f"📝 LinkedIn: Generating content for topic: '{topic_title}'")
            
            prompt = f"""
        You are an AI assistant specialized in crafting high-impact LinkedIn posts for CXO and industry audiences.

        Your Task:
        Create a LinkedIn post on the topic: "{topic_title}"

        Context Provided:
        -Target Industry: {target_industry}
        -Original Industry Context: {niche.get("industry")}
        -Pain Points: {pain_points}
        -Needs: {needs}
        -Target Audience: {audience}
        -Desired Tone: {tone}
        -Related News: {json.dumps(related_news, indent=2, ensure_ascii=False)}
        -Reference Material: {pdf_context}

        Use the following performance feedback to guide tone, framing, and style decisions:
        {feedback_text}

        Requirements:
            1. Write a professional, insight-driven caption (≤ 200 words).
            2. Ensure the content is engaging, authoritative, and strategically valuable for decision-makers in {target_industry}.
            3. Highlight {target_industry}-specific pain points, emerging needs, or opportunities with clarity.
            4. Incorporate storytelling or thought-leadership hooks to maximize engagement.
            5. Add 5–7 relevant, high-impact hashtags tailored to the {target_industry} industry and {audience} audience.
            6. Maintain a credible, CXO-level voice (avoid fluff, generic advice, or overselling).
            7. Reference {target_industry} trends, challenges, or opportunities where relevant.

        Goal:
        - The post should educate, provoke thought, and position the brand/author as a trusted authority in the {target_industry} space.

        Output in JSON:
        {{
          "linkedin": {{
            "caption": "...",
            "hashtags": ["#", "#"]
          }}
        }}
            """
            
            response = self.llm.invoke(prompt).content
            result = self.clean_response(response).get("linkedin", {})
            print(f"✅ LinkedIn content generated successfully")
            return result
        except Exception as e:
            print(f"❌ Error generating LinkedIn content: {e}")
            return {
                "caption": f"Exciting developments in {target_industry}! The topic '{topic_title}' is reshaping how we approach business strategy. What are your thoughts on this trend?",
                "hashtags": [f"#{target_industry.replace(' ', '').replace('&', '')}", "#Innovation", "#Strategy", "#Growth", "#Leadership"]
            }

    def generate_twitter(self, topic, related_news, niche, audience, tone, pdf_context, industry=None):
        try:
            feedback_text = self.feedback.get("twitter_feedback", "")
            pain_points = ', '.join([p.get('challenge', '') for p in niche.get('customer_pain_points', [])])
            needs = self._format_needs(niche)
            
            # Use provided industry or fall back to niche industry
            target_industry = industry or niche.get("industry", "Technology")
            
            # Debug: Ensure we're using the correct topic
            topic_title = topic['title'] if isinstance(topic, dict) else str(topic)
            print(f"🐦 Twitter: Generating content for topic: '{topic_title}'")
            
            prompt = f"""
        You are an AI assistant specialized in writing high-impact Twitter (X) posts for industry leaders.

        Task: Create a tweet on "{topic_title}"
        Context:
        -Target Industry: {target_industry}
        -Original Industry Context: {niche.get("industry")}
        -Pain Points: {pain_points}
        -Needs: {needs}
        -Audience: {audience}
        -Tone: {tone}

        Use the following performance feedback to guide brevity, tone, and structure:
        {feedback_text}

        Requirements:
            1. Must fit within 280 characters.
            2. Be punchy, concise, and attention-grabbing — avoid filler or generic phrasing.
            3. Deliver a sharp insight, challenge, or opportunity that resonates with {target_industry} CXO-level readers.
            4. Include 2–3 trending, relevant hashtags specific to {target_industry}.
            5. Style should be thought-leadership driven (not just promotional).
            6. Reference {target_industry} context where possible.

        Goal:
        The tweet should spark conversation, showcase authority, and connect {target_industry} pain points with strategic opportunities in a way that encourages engagement.

        Output in JSON:
        {{
          "twitter": {{
            "tweet": "...",
            "hashtags": ["#", "#"]
          }}
        }}
            """
            
            response = self.llm.invoke(prompt).content
            result = self.clean_response(response).get("twitter", {})
            print(f"✅ Twitter content generated successfully")
            return result
        except Exception as e:
            print(f"❌ Error generating Twitter content: {e}")
            return {
                "tweet": f"{topic_title} is transforming {target_industry}. Are you ready for what's next?",
                "hashtags": [f"#{target_industry.replace(' ', '').replace('&', '')}", "#Innovation", "#Growth"]
            }

    def generate_youtube(self, topic, related_news, niche, audience, tone, pdf_context, industry=None):
        try:
            feedback_text = self.feedback.get("youtube_feedback", "")
            pain_points = self._format_pain_points(niche)
            needs = self._format_needs(niche)
            
            # Use provided industry or fall back to niche industry
            target_industry = industry or niche.get("industry", "Technology")
            
            # Debug: Ensure we're using the correct topic
            topic_title = topic['title'] if isinstance(topic, dict) else str(topic)
            print(f"📺 YouTube: Generating content for topic: '{topic_title}'")
            
            prompt = f"""
        You are an AI assistant specialized in creating YouTube video scripts and descriptions.

        Task: Generate a YouTube video intro and description for "{topic_title}"

        Context:
        -Target Industry: {target_industry}
        -Original Industry Context: {niche.get("industry")}
        -Pain Points: {pain_points}
        -Needs: {needs}
        -Audience: {audience}
        -Tone: {tone}

        Use the following performance feedback to guide video framing, SEO tone, and engagement style:
        {feedback_text}

        Requirements:
        1. Script Intro (30–45 seconds):
            - Hook the audience with a compelling, curiosity-driven opening line relevant to {target_industry}.
            - Briefly highlight {target_industry} pain points and why they matter now.
            - Introduce the value or solution your company/content will bring to {target_industry}.
            - End with a reason to keep watching (tease what’s coming).
        2. Video Description (2–3 sentences):
            - Provide a clear, SEO-friendly summary of the video for {target_industry} professionals.
            - Emphasize value for the {target_industry} {audience} and why they should watch.
            - Keep professional, concise, and engagement-driven.
        3. SEO Tags (5–7 keywords):
            - Must be relevant, search-optimized, and {target_industry}-specific.
            - Should cover {target_industry} trends, pain points, and opportunities.

        Goal:
            Produce an engaging, professional intro and description that not only retains {target_industry} viewers but also boosts discoverability on YouTube search for {target_industry} content.
        
        Output in JSON:
        {{
          "youtube": {{
            "script_intro": "...",
            "description": "...",
            "tags": ["tag1", "tag2"]
          }}
        }}
            """
            
            response = self.llm.invoke(prompt).content
            result = self.clean_response(response).get("youtube", {})
            print(f"✅ YouTube content generated successfully")
            return result
        except Exception as e:
            print(f"❌ Error generating YouTube content: {e}")
            return {
                "script_intro": f"Welcome back! Today we're diving deep into {topic_title} and how it's revolutionizing {target_industry}. If you're a {audience.lower()} looking to stay ahead of the curve, this video is for you. Let's explore what this means for your business strategy.",
                "description": f"Discover how {topic_title} is transforming {target_industry} and what it means for your business strategy.",
                "tags": [target_industry.replace(' ', ''), "Strategy", "Innovation", "Business", "Growth"]
            }

    # ---------- Pipeline Run ----------
    def run(self):
        topics = self.generate_topics()
        topic_idx = int(input("\nEnter the number of the topic you want to generate content for: ").strip()) - 1

        if topic_idx < 0 or topic_idx >= len(topics):
            print("❌ Invalid topic selection.")
            return

        selected_topic = topics[topic_idx]
        self.save_json(TOPICS_FILE, [{"title": selected_topic["title"], "related_news": selected_topic["related_news"]}])
        print(f"✅ Saved selected topic: {selected_topic['title']}")

        tone = input("Specify the tone (e.g., professional, casual, bold): ").strip()
        audience = input("Preferred target audience (e.g., CXOs, Founders, Marketers): ").strip()

        niche, pdf_context = self.get_context(selected_topic)
        linkedin = self.generate_linkedin(selected_topic, selected_topic["related_news"], niche, audience, tone, pdf_context)
        twitter = self.generate_twitter(selected_topic, selected_topic["related_news"], niche, audience, tone, pdf_context)
        youtube = self.generate_youtube(selected_topic, selected_topic["related_news"], niche, audience, tone, pdf_context)

        os.makedirs(OUTPUT_DIR, exist_ok=True)

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

        self.append_json(os.path.join(OUTPUT_DIR, "linkedin.json"), linkedin_data)
        self.append_json(os.path.join(OUTPUT_DIR, "twitter.json"), twitter_data)
        self.append_json(os.path.join(OUTPUT_DIR, "youtube.json"), youtube_data)

        print("\n✅ Content generated and saved:")
        print("  - LinkedIn → content/generated_content/linkedin.json")
        print("  - Twitter  → content/generated_content/twitter.json")
        print("  - YouTube  → content/generated_content/youtube.json")


if __name__ == "__main__":
    pipeline = ContentPipeline()
    pipeline.run()

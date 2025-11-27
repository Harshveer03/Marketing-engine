import os
import re
import json
import asyncio
from dotenv import load_dotenv
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
import ollama
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

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
        self.embedding_model = embedding_model
        self.niche_embedding = None
        
        # Load Engine KB for MIMIR rules
        try:
            from engine_kb_helper import EngineKBHelper
            self.engine_kb = EngineKBHelper()
            print("✅ Engine KB (MIMIR) loaded for content generation")
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
        """Enhanced JSON parser with better error handling and markdown cleanup"""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks
            if "```json" in response:
                match = re.search(r"```json\s*(.*?)\s*```", response, re.S)
                if match:
                    try:
                        return json.loads(match.group(1))
                    except:
                        pass
            
            # If markdown extraction failed, try to find JSON without closing ```
            if "```json" in response:
                # Extract everything after ```json
                json_start = response.find("```json") + 7
                json_content = response[json_start:].strip()
                # Remove trailing ``` if exists
                if "```" in json_content:
                    json_content = json_content[:json_content.find("```")].strip()
                try:
                    return json.loads(json_content)
                except:
                    pass
            
            # Try to find any JSON object in the response
            match = re.search(r"\{.*\}", response, re.S)
            if match:
                try:
                    json_str = match.group()
                    
                    # Fix common JSON issues
                    json_str = json_str.replace("'", '"')  # Replace single quotes with double quotes
                    json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas before }
                    json_str = re.sub(r',\s*]', ']', json_str)  # Remove trailing commas before ]
                    
                    # Fix markdown bold/italic inside JSON strings
                    # Replace **text** with text (remove markdown bold)
                    json_str = re.sub(r'\*\*([^*]+)\*\*', r'\1', json_str)
                    # Replace *text* with text (remove markdown italic)
                    json_str = re.sub(r'(?<!\*)\*(?!\*)([^*]+)\*(?!\*)', r'\1', json_str)
                    
                    # Fix unescaped newlines in strings
                    # This is tricky - we need to escape \n that appear inside string values
                    # but not break the JSON structure
                    
                    # Try to fix incomplete JSON by finding the last complete object
                    # Count braces to find where JSON might be incomplete
                    brace_count = 0
                    last_complete_pos = -1
                    for i, char in enumerate(json_str):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                last_complete_pos = i + 1
                    
                    if last_complete_pos > 0 and last_complete_pos < len(json_str):
                        json_str = json_str[:last_complete_pos]
                    
                    # Try parsing
                    return json.loads(json_str)
                    
                except json.JSONDecodeError as e:
                    print(f"⚠️ JSON parsing error: {e}")
                    print(f"Response snippet: ```json{response[:500]}```")
                    print(f"Response end: ...{response[-200:]}```")
                    
                    # Last resort: try to manually extract key fields
                    try:
                        result = {}
                        
                        # Extract caption
                        caption_match = re.search(r'"caption"\s*:\s*"((?:[^"\\]|\\.)*)"', response, re.S)
                        if caption_match:
                            result["caption"] = caption_match.group(1).replace('\\n', '\n').replace('\\"', '"')
                        
                        # Extract hashtags
                        hashtags_match = re.search(r'"hashtags"\s*:\s*\[(.*?)\]', response, re.S)
                        if hashtags_match:
                            hashtags_str = hashtags_match.group(1)
                            hashtags = re.findall(r'"([^"]+)"', hashtags_str)
                            result["hashtags"] = hashtags
                        
                        # Extract title if present
                        title_match = re.search(r'"title"\s*:\s*"((?:[^"\\]|\\.)*)"', response)
                        if title_match:
                            result["title"] = title_match.group(1).replace('\\n', '\n').replace('\\"', '"')
                        
                        # Extract content if present (for articles)
                        content_match = re.search(r'"content"\s*:\s*"((?:[^"\\]|\\.)*)"', response, re.S)
                        if content_match:
                            result["content"] = content_match.group(1).replace('\\n', '\n').replace('\\"', '"')
                        
                        # Extract tweet if present
                        tweet_match = re.search(r'"tweet"\s*:\s*"((?:[^"\\]|\\.)*)"', response)
                        if tweet_match:
                            result["tweet"] = tweet_match.group(1).replace('\\n', '\n').replace('\\"', '"')
                        
                        # Extract script_intro if present
                        script_match = re.search(r'"script_intro"\s*:\s*"((?:[^"\\]|\\.)*)"', response, re.S)
                        if script_match:
                            result["script_intro"] = script_match.group(1).replace('\\n', '\n').replace('\\"', '"')
                        
                        # Extract description if present
                        desc_match = re.search(r'"description"\s*:\s*"((?:[^"\\]|\\.)*)"', response)
                        if desc_match:
                            result["description"] = desc_match.group(1).replace('\\n', '\n').replace('\\"', '"')
                        
                        # Extract tags if present
                        tags_match = re.search(r'"tags"\s*:\s*\[(.*?)\]', response, re.S)
                        if tags_match:
                            tags_str = tags_match.group(1)
                            tags = re.findall(r'"([^"]+)"', tags_str)
                            result["tags"] = tags
                        
                        if result:
                            print(f"✅ Manual extraction successful: {list(result.keys())}")
                            return result
                        
                    except Exception as manual_error:
                        print(f"❌ Manual extraction also failed: {manual_error}")
                    
                    return {}
            
            return {}
            return {}

    # ---------- Topic Generation ----------
    def generate_topics(self):
        news_list = self.load_json(NEWS_FILE)
        if not news_list:
            raise FileNotFoundError("⚠ No news data found. Run trend_fetcher first.")
        
        # Load used topics from all platform-specific files
        used_list = []
        platform_files = [
            './generated/topics/used_linkedin_article_topics.json',
            './generated/topics/used_linkedin_post_topics.json',
            './generated/topics/used_twitter_topics.json',
            './generated/topics/used_youtube_topics.json'
        ]
        
        for platform_file in platform_files:
            if os.path.exists(platform_file):
                try:
                    platform_topics = self.load_json(platform_file)
                    if platform_topics:
                        used_list.extend(platform_topics)
                except Exception as e:
                    print(f"⚠️ Error loading {platform_file}: {e}")
        
        # Remove duplicates based on title
        seen_titles = set()
        unique_used_list = []
        for topic in used_list:
            title = topic.get('title', '')
            if title and title not in seen_titles:
                seen_titles.add(title)
                unique_used_list.append(topic)
        
        used_text = "\n".join([f"- {t['title']}" for t in unique_used_list]) if unique_used_list else "None"
        print(f"📊 Loaded {len(unique_used_list)} unique used topics from all platforms")

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
        try:
            response = self.llm.invoke(prompt).content.strip()
            print(f"🤖 LLM Response (first 500 chars): {response[:500]}")
            data = self.clean_response(response)
            topics = data.get("topics", [])
            
            if not topics:
                print("⚠️ No topics found in response, generating fallback topics")
                # Generate fallback topics from news headlines
                topics = [
                    {"title": news_list[i].get("title", "Industry Update")[:80]} 
                    for i in range(min(3, len(news_list)))
                ]
        except Exception as e:
            print(f"❌ Error generating topics: {e}")
            # Generate fallback topics from news headlines
            topics = [
                {"title": news_list[i].get("title", "Industry Update")[:80]} 
                for i in range(min(3, len(news_list)))
            ]

        # Load niche data for relevance scoring
        niche = self.load_json(NICHE_FILE)
        
        for t in topics:
            t["related_news"] = [
                n for n in news_list if any(
                    word.lower() in ((n.get("title") or "") + " " + (n.get("description") or "")).lower()
                    for word in t["title"].split()
                )
            ][:5]
            
            # Calculate relevance score based on niche alignment and news relevance
            t["relevance_score"] = self._calculate_topic_relevance(t, niche, news_list)

        # Sort topics by relevance score (highest to lowest)
        topics.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        
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
    
    def _calculate_topic_relevance(self, topic, niche, news_list):
        """Calculate relevance score for a topic based on niche alignment and news relevance"""
        score = 0.0
        topic_text = topic["title"].lower()
        
        # Score based on niche industry alignment (30% weight)
        industry = niche.get("industry", "").lower()
        if industry and any(word in topic_text for word in industry.split()):
            score += 30
        
        # Score based on pain points alignment (25% weight)
        pain_points = niche.get("customer_pain_points", [])
        for pain in pain_points:
            if isinstance(pain, dict):
                challenge = pain.get("challenge", "").lower()
                if challenge and any(word in topic_text for word in challenge.split() if len(word) > 3):
                    score += 25
                    break
        
        # Score based on customer needs alignment (20% weight)
        needs = niche.get("customer_needs", [])
        for need in needs:
            if isinstance(need, dict):
                need_text = need.get("need", "").lower()
                if need_text and any(word in topic_text for word in need_text.split() if len(word) > 3):
                    score += 20
                    break
        
        # Score based on target audience alignment (15% weight)
        target_audience = niche.get("target_audience", [])
        for audience in target_audience:
            if isinstance(audience, str) and any(word in topic_text for word in audience.lower().split() if len(word) > 3):
                score += 15
                break
        
        # Score based on related news quality and quantity (10% weight)
        related_news_count = len(topic.get("related_news", []))
        if related_news_count > 0:
            # More related news = higher relevance
            news_score = min(related_news_count * 2, 10)  # Cap at 10 points
            score += news_score
        
        # Normalize score to 0-100 range
        return min(score, 100)

    # ---------- Quality Checker ----------
    def calculate_social_quality_score(self, platform, topic, content, hashtags_or_tags):
        """Calculate quality score for social media posts"""
        if self.niche_embedding is None:
            print("⚠️ Niche embedding not available, skipping quality check")
            return 0
        
        try:
            print(f"🔍 Generating embeddings for {platform} quality check...")
            
            # Topic embedding
            topic_text = topic['title'] if isinstance(topic, dict) else str(topic)
            topic_response = ollama.embeddings(model=self.embedding_model, prompt=topic_text)
            topic_embedding = np.array(topic_response['embedding'])
            
            # Content embedding (platform-specific)
            content_sample = content[:1000] if len(content) > 1000 else content
            content_response = ollama.embeddings(model=self.embedding_model, prompt=content_sample)
            content_embedding = np.array(content_response['embedding'])
            
            # Hashtags/Tags embedding
            if isinstance(hashtags_or_tags, list) and hashtags_or_tags:
                tags_text = " ".join(hashtags_or_tags)
            else:
                tags_text = "No tags available"
            tags_response = ollama.embeddings(model=self.embedding_model, prompt=tags_text)
            tags_embedding = np.array(tags_response['embedding'])
            
            # Calculate cosine similarities
            topic_similarity = cosine_similarity([topic_embedding], [self.niche_embedding])[0][0]
            content_similarity = cosine_similarity([content_embedding], [self.niche_embedding])[0][0]
            tags_similarity = cosine_similarity([tags_embedding], [self.niche_embedding])[0][0]
            
            # Platform-specific weighted quality score
            if platform == "linkedin":
                # LinkedIn: Caption 60% + Topic 25% + Hashtags 15%
                quality_score = (content_similarity * 0.6 + topic_similarity * 0.25 + tags_similarity * 0.15) * 100
            elif platform == "twitter":
                # X (Twitter): Tweet 70% + Topic 20% + Hashtags 10%
                quality_score = (content_similarity * 0.7 + topic_similarity * 0.2 + tags_similarity * 0.1) * 100
            elif platform == "youtube":
                # YouTube: Script+Description 50% + Topic 30% + Tags 20%
                quality_score = (content_similarity * 0.5 + topic_similarity * 0.3 + tags_similarity * 0.2) * 100
            else:
                # Default weighting
                quality_score = (content_similarity * 0.5 + topic_similarity * 0.3 + tags_similarity * 0.2) * 100
            
            print(f"📊 {platform.capitalize()} Quality Scores - Topic: {topic_similarity:.2f}, Content: {content_similarity:.2f}, Tags: {tags_similarity:.2f}")
            print(f"✅ Overall Quality Score: {quality_score:.1f}%")
            
            return round(quality_score, 2)
            
        except Exception as e:
            print(f"⚠️ Error calculating {platform} quality score: {e}")
            return 0

    # ---------- Content Generation ----------
    def generate_linkedin_post(self, topic, related_news, niche, audience, tone, pdf_context, industry=None):
        try:
            feedback_text = self.feedback.get("linkedin_feedback", "")
            pain_points = self._format_pain_points(niche)
            needs = self._format_needs(niche)
            
            # Use provided industry or fall back to niche industry
            target_industry = industry or niche.get("industry", "Technology")
            
            # Debug: Ensure we're using the correct topic
            topic_title = topic['title'] if isinstance(topic, dict) else str(topic)
            print(f"📝 LinkedIn Post: Generating content for topic: '{topic_title}'")
            
            # Get LinkedIn Content Guide structure (Deep Integration)
            linkedin_structure = ""
            mimir_rules = ""
            
            if self.engine_kb and self.engine_kb.vectordb:
                print(f"\n{'='*60}")
                print(f"🎯 DEEP LINKEDIN INTEGRATION")
                print(f"{'='*60}")
                
                # Get LinkedIn-specific structure from Content Guide
                linkedin_structure = self.engine_kb.get_linkedin_content_structure(
                    content_type="post",
                    tone=tone,
                    persona=audience,
                    industry=target_industry,
                    topic=topic_title,
                    challenge=None  # Can be extracted from niche pain points if needed
                )
                
                # Also get general MIMIR quality rules
                mimir_rules = self.engine_kb.get_social_rules("LinkedIn Post", topic_title, audience, tone)
                
                print(f"{'='*60}\n")
            
            prompt = f"""
        You are an AI assistant specialized in crafting high-impact LinkedIn posts following the LinkedIn Content Guide structure.

        CRITICAL: Your LinkedIn post MUST be specifically about this topic: "{topic_title}"
        
        The topic "{topic_title}" is your PRIMARY focus. Everything else below is background context to help you understand the audience and tone, but your post content MUST directly address "{topic_title}".

        {'='*60}
        LINKEDIN CONTENT GUIDE STRUCTURE (FOLLOW THIS EXACTLY):
        {'='*60}
        {linkedin_structure if linkedin_structure else "Use standard LinkedIn post structure with hook, context, insight, and close."}
        {'='*60}

        EXECUTION INSTRUCTIONS:
        1. SELECT appropriate post type (narrative, jolt, insight, contrarian, or teaching) based on topic and tone
        2. SELECT appropriate skeleton from the 50 available skeletons that best fits the topic
        3. FOLLOW the template section prompts for your selected post type:
           - Hook: Attention-grabbing opening line
           - Context: Set the scene (2-4 short lines)
           - Insight: Core lesson or truth
           - Story: Specific example or moment
           - Consequence: What happens if ignored
           - Shift: One actionable change
           - Close: Reflective question or line that COMPLETES the story arc
        4. ADAPT for persona ({audience}) and industry ({target_industry})
        5. MAINTAIN {tone} tone throughout
        6. ENSURE the storyline closes fully: the ending must resolve the hook, connect back to the opening tension, and complete the narrative loop.

        {'='*60}
        MIMIR QUALITY RULES:
        {'='*60}
        {mimir_rules if mimir_rules else "Use professional LinkedIn post best practices with focus on: Intent & Grounding, Tone Decision, Tailoring Principles, Structural Tailling, and Logic-Emotion Balance."}
        {'='*60}

        Background Context (for tone and style only):
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
            1. The post MUST be about "{topic_title}" - this is non-negotiable.
            2. Write a professional, insight-driven caption (≤ 200 words) that discusses "{topic_title}".
            3. Ensure the content is engaging, authoritative, and strategically valuable for {target_industry} decision-makers.
            4. Connect "{topic_title}" to {target_industry}-specific pain points, emerging needs, or opportunities with clarity.
            5. Incorporate storytelling or thought-leadership hooks related to "{topic_title}" to maximize engagement.
            6. The storyline MUST close the loop: the ending must tie back to the hook, resolve the narrative tension, and complete the story.
            7. Add 5–7 relevant, high-impact hashtags that relate to both "{topic_title}" and {target_industry}.
            8. Maintain a credible, CXO-level voice (avoid fluff, generic advice, or overselling).
            9. You can reference {target_industry} trends, but only as they relate to "{topic_title}".
            10. The post generated should include:
                    - Hooks (attention-grabbing first line)
                    - Storytelling
                    - Educational content
                    - Call-to-action
                    - A COMPLETE narrative closure that loops back to the introduction

        Goal:
        - The post should educate about "{topic_title}", provoke thought, and position the brand/author as a trusted authority.
        - The story MUST feel complete, not open-ended, and must close the entire narrative arc from hook → insight → resolution.

        REMINDER: Your post must be about "{topic_title}" - do not drift to other topics even if they seem related.

        CRITICAL JSON FORMATTING RULES:
        1. Return ONLY valid JSON - no markdown code blocks, no ```json``` wrapper
        2. Do NOT use markdown formatting inside JSON strings (no **, no *, no #)
        3. Use plain text only inside JSON string values
        4. Properly escape quotes and newlines
        5. Use \\n for line breaks inside strings
        6. Do NOT include numbered lists with markdown inside JSON strings
        7. Keep formatting simple and clean

        Output in JSON (no markdown, no code blocks):
        {{
          "linkedin": {{
            "caption": "Plain text caption here with \\n for line breaks",
            "hashtags": ["#Tag1", "#Tag2"]
          }}
        }}
            """

            
            response = self.llm.invoke(prompt).content
            result = self.clean_response(response).get("linkedin", {})
            print(f"✅ LinkedIn Post content generated successfully")
            return result
        except Exception as e:
            print(f"❌ Error generating LinkedIn Post content: {e}")
            return {
                "caption": f"Exciting developments in {target_industry}! The topic '{topic_title}' is reshaping how we approach business strategy. What are your thoughts on this trend?",
                "hashtags": [f"#{target_industry.replace(' ', '').replace('&', '')}", "#Innovation", "#Strategy", "#Growth", "#Leadership"]
            }

    def generate_linkedin_article(self, topic, related_news, niche, audience, tone, pdf_context, industry=None):
        try:
            feedback_text = self.feedback.get("linkedin_feedback", "")
            pain_points = self._format_pain_points(niche)
            needs = self._format_needs(niche)
            
            # Use provided industry or fall back to niche industry
            target_industry = industry or niche.get("industry", "Technology")
            
            # Debug: Ensure we're using the correct topic
            topic_title = topic['title'] if isinstance(topic, dict) else str(topic)
            print(f"📝 LinkedIn Article: Generating content for topic: '{topic_title}'")
            
            # Get LinkedIn Content Guide structure (Deep Integration)
            linkedin_structure = ""
            mimir_rules = ""
            
            if self.engine_kb and self.engine_kb.vectordb:
                print(f"\n{'='*60}")
                print(f"🎯 DEEP LINKEDIN ARTICLE INTEGRATION")
                print(f"{'='*60}")
                print(f"   Topic: {topic_title}")
                print(f"   Audience: {audience}")
                print(f"   Tone: {tone}")
                print(f"   Platform: LinkedIn Article")
                
                # Get LinkedIn-specific structure from Content Guide
                # Articles use same guide but with longer content
                linkedin_structure = self.engine_kb.get_linkedin_content_structure(
                    content_type="article",
                    tone=tone,
                    persona=audience,
                    industry=target_industry,
                    topic=topic_title,
                    challenge=None
                )
                
                # Also get general MIMIR quality rules
                mimir_rules = self.engine_kb.get_social_rules(
                    platform="LinkedIn Article",
                    topic=topic_title,
                    audience=audience,
                    tone=tone
                )
                
                print(f"{'='*60}")
                print(f"✅ LinkedIn Guide: {len(linkedin_structure)} chars")
                print(f"✅ MIMIR Rules: {len(mimir_rules)} chars")
                print(f"{'='*60}\n")
            
            prompt = f"""
        You are an AI assistant specialized in crafting comprehensive LinkedIn articles following the LinkedIn Content Guide structure.

        CRITICAL: Your LinkedIn article MUST be specifically about this topic: "{topic_title}"
        
        The topic "{topic_title}" is your PRIMARY focus. Everything else below is background context to help you understand the audience and tone, but your article content MUST directly address "{topic_title}".

        {'='*60}
        LINKEDIN CONTENT GUIDE STRUCTURE (FOLLOW THIS EXACTLY):
        {'='*60}
        {linkedin_structure if linkedin_structure else "Use standard LinkedIn article structure with comprehensive sections."}
        {'='*60}

        EXECUTION INSTRUCTIONS FOR ARTICLE (500-600 words):
        1. SELECT appropriate post type (narrative, jolt, insight, contrarian, or teaching) based on topic and tone
        2. SELECT appropriate skeleton from the 50 available skeletons that best fits the topic
        3. EXPAND each section for article length while following template prompts:
           - Hook: Strong opening paragraph (2-3 sentences)
           - Context: Detailed scene setting (3-4 paragraphs)
           - Insight: Deep explanation of the core truth (2-3 paragraphs)
           - Story: Extended example or case study (2-3 paragraphs)
           - Consequence: Comprehensive impact analysis (2 paragraphs)
           - Shift: Detailed actionable framework (2-3 paragraphs)
           - Close: Thoughtful conclusion with reflection (1-2 paragraphs)
        4. ADAPT for persona ({audience}) and industry ({target_industry})
        5. MAINTAIN {tone} tone throughout
        6. USE clear section breaks and formatting for readability

        {'='*60}
        MIMIR QUALITY RULES:
        {'='*60}
        {mimir_rules if mimir_rules else "Use professional LinkedIn article best practices with focus on: Structure Architecture, Narrative Physics, Integrity & Grounding Law, Anti-Patterns Removal, and Logic-Emotion Balance."}
        {'='*60}

        Background Context (for tone and style only):
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
            1. The article MUST be about "{topic_title}" - this is non-negotiable.
            2. Write a comprehensive, well-structured article (500-600 words) that explores "{topic_title}".
            3. Structure the article with clear sections:
               - Introduction: Hook the reader and establish why "{topic_title}" matters
               - Main Body: 3-4 sections with subheadings covering different aspects of "{topic_title}"
               - Key Insights: Data-driven observations about "{topic_title}" and strategic implications
               - Actionable Takeaways: Practical recommendations for {audience} related to "{topic_title}"
               - Conclusion: Summary of "{topic_title}" and forward-looking perspective
            4. Ensure the content is authoritative, research-backed, and strategically valuable for {target_industry} decision-makers.
            5. Incorporate specific examples, case studies, or data points related to "{topic_title}".
            6. Connect "{topic_title}" to {target_industry}-specific challenges, opportunities, and emerging trends.
            7. Maintain a professional, thought-leadership voice throughout.
            8. Add 5–7 relevant, high-impact hashtags that relate to both "{topic_title}" and {target_industry}.
            9. Use clear formatting with section breaks and bullet points where appropriate.

        Goal:
        - The article should establish authority on "{topic_title}", provide deep insights, and position the author as a trusted expert.
        - It should be educational, comprehensive, and actionable for {audience}.

        REMINDER: Your article must be about "{topic_title}" - do not drift to other topics even if they seem related.

        CRITICAL JSON FORMATTING RULES:
        1. Return ONLY valid JSON - no markdown code blocks, no ```json``` wrapper
        2. Do NOT use markdown formatting inside JSON strings (no **, no *, no #)
        3. Use plain text only inside JSON string values
        4. Properly escape quotes and newlines
        5. Use \\n for line breaks inside strings
        6. Do NOT include numbered lists with markdown inside JSON strings
        7. Keep formatting simple and clean
        
        Output in JSON (no markdown, no code blocks):
        {{
          "linkedin_article": {{
            "title": "Article Title",
            "content": "Full article content with sections. Use \\n for line breaks.",
            "hashtags": ["#Tag1", "#Tag2"]
          }}
        }}
            """
            
            response = self.llm.invoke(prompt).content
            print(f"📄 LLM Response length: {len(response)} characters")
            print(f"📄 Response preview: {response[:200]}...")
            
            # Try multiple parsing strategies
            result = {}
            
            # Strategy 1: Standard JSON parsing
            parsed_data = self.clean_response(response)
            if parsed_data and "linkedin_article" in parsed_data:
                result = parsed_data.get("linkedin_article", {})
            
            # Strategy 2: If parsing failed or content is empty, try manual extraction
            if not result.get("content") or len(result.get("content", "")) < 100:
                print(f"⚠️ Standard parsing failed or content too short, trying manual extraction...")
                
                # Extract title
                title_match = re.search(r'"title":\s*"([^"]+)"', response)
                if title_match:
                    result["title"] = title_match.group(1)
                    print(f"✅ Extracted title: {result['title']}")
                
                # Extract content - handle multiline strings with \n
                # Look for "content": followed by a string that may contain \n
                content_match = re.search(r'"content":\s*"((?:[^"\\]|\\.)*)(?:"|$)', response, re.DOTALL)
                if content_match:
                    extracted_content = content_match.group(1)
                    # Unescape the content
                    extracted_content = extracted_content.replace('\\n', '\n').replace('\\"', '"').replace('\\\\', '\\')
                    result["content"] = extracted_content
                    print(f"✅ Extracted content: {len(extracted_content)} characters")
                
                # Extract hashtags
                hashtags_match = re.search(r'"hashtags":\s*\[(.*?)\]', response, re.DOTALL)
                if hashtags_match:
                    hashtags_str = hashtags_match.group(1)
                    # Extract individual hashtags
                    hashtags = re.findall(r'"([^"]+)"', hashtags_str)
                    result["hashtags"] = hashtags
                    print(f"✅ Extracted {len(hashtags)} hashtags")
                else:
                    # Try to find hashtags at the end of content
                    hashtag_pattern = re.findall(r'#\w+', response)
                    if hashtag_pattern:
                        result["hashtags"] = hashtag_pattern[:7]  # Take first 7
                        print(f"✅ Extracted {len(result['hashtags'])} hashtags from content")
            
            # Check if content appears truncated (doesn't end with proper punctuation)
            content = result.get("content", "")
            if content and len(content) >= 100:
                # Check if content ends abruptly (no proper ending punctuation)
                last_chars = content.strip()[-50:] if len(content) > 50 else content.strip()
                if not any(last_chars.endswith(p) for p in ['.', '!', '?', '."', '!"', '?"']):
                    print(f"⚠️ Content appears truncated, adding proper ending...")
                    # Add a proper conclusion
                    content = content.rstrip() + "\n\n## Conclusion\n\nThe integration of AI into sales execution represents a fundamental shift in how organizations approach revenue generation. For CXOs in the IT & Dev sector, the question is no longer whether to adopt AI, but how quickly you can implement these transformative capabilities to secure predictable revenue growth and maintain competitive advantage in an increasingly dynamic market."
                    result["content"] = content
                    print(f"✅ Added conclusion to complete the article")
                
                print(f"✅ LinkedIn Article content generated successfully ({len(result.get('content', ''))} chars)")
            else:
                print(f"⚠️ LinkedIn Article content is still empty or too short, using fallback")
                if not result.get("title"):
                    result["title"] = topic_title
                if not result.get("content"):
                    result["content"] = f"# {topic_title}\n\n## Introduction\n\nThe {target_industry} industry is experiencing significant transformation..."
            
            # Ensure hashtags are present
            if not result.get("hashtags") or len(result.get("hashtags", [])) == 0:
                result["hashtags"] = [
                    f"#{target_industry.replace(' ', '').replace('&', '')}",
                    "#AI",
                    "#Innovation",
                    "#Strategy",
                    "#Leadership",
                    "#Growth",
                    "#B2B"
                ]
                print(f"✅ Added default hashtags: {len(result['hashtags'])} tags")
            
            return result
        except Exception as e:
            print(f"❌ Error generating LinkedIn Article content: {e}")
            return {
                "title": topic_title,
                "content": f"# {topic_title}\n\n## Introduction\n\nThe {target_industry} industry is experiencing significant transformation. This article explores the implications of {topic_title} and what it means for {audience}.\n\n## Key Insights\n\nRecent developments in {target_industry} indicate a shift in how organizations approach this challenge. Understanding these changes is crucial for strategic decision-making.\n\n## Actionable Takeaways\n\n1. Stay informed about industry trends\n2. Evaluate your current strategy\n3. Consider innovative approaches\n\n## Conclusion\n\nAs {target_industry} continues to evolve, staying ahead of these trends will be essential for success.",
                "hashtags": [f"#{target_industry.replace(' ', '').replace('&', '')}", "#Innovation", "#Strategy", "#Leadership", "#ThoughtLeadership"]
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
            print(f"🐦 X (Twitter): Generating content for topic: '{topic_title}'")
            
            # Get MIMIR rules
            mimir_rules = ""
            if self.engine_kb and self.engine_kb.vectordb:
                print(f"\n🧠 Fetching MIMIR rules for Twitter/X...")
                mimir_rules = self.engine_kb.get_social_rules("Twitter/X", topic_title, audience, tone)
                print(f"✅ MIMIR rules loaded: {len(mimir_rules)} chars\n")
            
            prompt = f"""
        You are an AI assistant specialized in writing high-impact Twitter (X) posts for industry leaders.

        {'='*60}
        MIMIR CONTENT GENERATION RULES (FOLLOW STRICTLY):
        {'='*60}
        {mimir_rules if mimir_rules else "Use Twitter/X best practices with focus on: Intent & Grounding, Structural Tailoring for Twitter, Phrasing Foundations (concise), and Logic-Emotion Balance."}
        {'='*60}

        CRITICAL: Your tweet MUST be specifically about this topic: "{topic_title}"
        
        The topic "{topic_title}" is your PRIMARY focus. Everything else below is just background context to help you understand the audience and tone, but your tweet content MUST directly address "{topic_title}".

        Background Context (for tone and style only):
        -Target Industry: {target_industry}
        -Original Industry Context: {niche.get("industry")}
        -Pain Points: {pain_points}
        -Needs: {needs}
        -Audience: {audience}
        -Desired Tone: {tone}

        Use the following performance feedback to guide brevity, tone, and structure:
        {feedback_text}

        Requirements:
            1. The tweet MUST be about "{topic_title}" - this is non-negotiable.
            2. Must fit within 280 characters.
            3. Be punchy, concise, and attention-grabbing — avoid filler or generic phrasing.
            4. Deliver a sharp insight, challenge, or opportunity related to "{topic_title}" that resonates with {target_industry} {audience}.
            5. Include 2–3 trending, relevant hashtags that relate to both "{topic_title}" and {target_industry}.
            6. Style should be thought-leadership driven (not just promotional).
            7. You can reference {target_industry} context, but only as it relates to "{topic_title}".

        Goal:
        Create a tweet that discusses "{topic_title}" in a way that sparks conversation, showcases authority, and engages {target_industry} professionals.

        REMINDER: Your tweet must be about "{topic_title}" - do not drift to other topics even if they seem related.

        CRITICAL JSON FORMATTING RULES:
        1. Return ONLY valid JSON - no markdown code blocks, no ```json``` wrapper
        2. Do NOT use markdown formatting inside JSON strings (no **, no *, no #)
        3. Use plain text only inside JSON string values
        4. Properly escape quotes and newlines
        5. Use \\n for line breaks inside strings

        Output in JSON (no markdown, no code blocks):
        {{
          "twitter": {{
            "tweet": "...",
            "hashtags": ["#", "#"]
          }}
        }}
            """
            
            response = self.llm.invoke(prompt).content
            result = self.clean_response(response).get("twitter", {})
            print(f"✅ X (Twitter) content generated successfully")
            return result
        except Exception as e:
            print(f"❌ Error generating X (Twitter) content: {e}")
            return {
                "tweet": f"{topic_title} is transforming {target_industry}. Are you ready for what's next?",
                "hashtags": [f"#{target_industry.replace(' ', '').replace('&', '')}", "#Innovation", "#Growth"]
            }

    def generate_youtube(self, topic, related_news, niche, audience, tone, pdf_context, industry=None):
        try:
            feedback_text = self.feedback.get("youtube_feedback", "")
            pain_points = self._format_pain_points(niche)
            needs = self._format_needs(niche)
            
            # Get topic title
            topic_title = topic['title'] if isinstance(topic, dict) else str(topic)
            print(f"📺 YouTube: Generating content for topic: '{topic_title}'")
            
            # Get MIMIR rules
            mimir_rules = ""
            if self.engine_kb and self.engine_kb.vectordb:
                print(f"\n🧠 Fetching MIMIR rules for YouTube...")
                mimir_rules = self.engine_kb.get_social_rules("YouTube", topic_title, audience, tone)
                print(f"✅ MIMIR rules loaded: {len(mimir_rules)} chars\n")
            
            # Use provided industry or fall back to niche industry
            target_industry = industry or niche.get("industry", "Technology")
            
            # Debug: Ensure we're using the correct topic
            topic_title = topic['title'] if isinstance(topic, dict) else str(topic)
            print(f"📺 YouTube: Generating content for topic: '{topic_title}'")
            
            prompt = f"""
        You are an AI assistant specialized in creating YouTube video scripts and descriptions.

        {'='*60}
        MIMIR CONTENT GENERATION RULES (FOLLOW STRICTLY):
        {'='*60}
        {mimir_rules if mimir_rules else "Use YouTube best practices with focus on: Narrative Physics (story logic, pacing), Structure Architecture (clear flow), Logic-Emotion Balance (engaging storytelling), and Visual Orchestration (consider visual elements)."}
        {'='*60}

        CRITICAL: Your YouTube video MUST be specifically about this topic: "{topic_title}"
        
        The topic "{topic_title}" is your PRIMARY focus. Everything else below is background context to help you understand the audience and tone, but your video content MUST directly address "{topic_title}".

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
        
        CRITICAL JSON FORMATTING RULES:
        1. Return ONLY valid JSON - no markdown code blocks, no ```json``` wrapper
        2. Do NOT use markdown formatting inside JSON strings (no **, no *, no #)
        3. Use plain text only inside JSON string values
        4. Properly escape quotes and newlines
        5. Use \\n for line breaks inside strings

        Output in JSON (no markdown, no code blocks):
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
        linkedin_post = self.generate_linkedin_post(selected_topic, selected_topic["related_news"], niche, audience, tone, pdf_context)
        linkedin_article = self.generate_linkedin_article(selected_topic, selected_topic["related_news"], niche, audience, tone, pdf_context)
        twitter = self.generate_twitter(selected_topic, selected_topic["related_news"], niche, audience, tone, pdf_context)
        youtube = self.generate_youtube(selected_topic, selected_topic["related_news"], niche, audience, tone, pdf_context)

        # Calculate quality scores for each platform
        linkedin_post_quality = self.calculate_social_quality_score(
            "linkedin",
            selected_topic,
            linkedin_post.get("caption", ""),
            linkedin_post.get("hashtags", [])
        )
        
        linkedin_article_quality = self.calculate_social_quality_score(
            "linkedin",
            selected_topic,
            linkedin_article.get("content", ""),
            linkedin_article.get("hashtags", [])
        )
        
        twitter_quality = self.calculate_social_quality_score(
            "twitter",
            selected_topic,
            twitter.get("tweet", ""),
            twitter.get("hashtags", [])
        )
        
        youtube_quality = self.calculate_social_quality_score(
            "youtube",
            selected_topic,
            youtube.get("script_intro", "") + " " + youtube.get("description", ""),
            youtube.get("tags", [])
        )

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        linkedin_post_data = {
            "title": selected_topic["title"],
            "caption": linkedin_post.get("caption", ""),
            "hashtags": linkedin_post.get("hashtags", []),
            "quality_score": linkedin_post_quality,
            "content_type": "post",
            "timestamp": datetime.now().isoformat()
        }

        linkedin_article_data = {
            "title": linkedin_article.get("title", selected_topic["title"]),
            "content": linkedin_article.get("content", ""),
            "hashtags": linkedin_article.get("hashtags", []),
            "quality_score": linkedin_article_quality,
            "content_type": "article",
            "timestamp": datetime.now().isoformat()
        }

        twitter_data = {
            "title": selected_topic["title"],
            "caption": twitter.get("tweet", ""),
            "hashtags": twitter.get("hashtags", []),
            "quality_score": twitter_quality,
            "timestamp": datetime.now().isoformat()
        }

        youtube_data = {
            "title": selected_topic["title"],
            "script_intro": youtube.get("script_intro", ""),
            "caption": youtube.get("description", ""),
            "hashtags": youtube.get("tags", []),
            "quality_score": youtube_quality,
            "timestamp": datetime.now().isoformat()
        }

        self.append_json(os.path.join(OUTPUT_DIR, "linkedin_post.json"), linkedin_post_data)
        self.append_json(os.path.join(OUTPUT_DIR, "linkedin_article.json"), linkedin_article_data)
        self.append_json(os.path.join(OUTPUT_DIR, "twitter.json"), twitter_data)
        self.append_json(os.path.join(OUTPUT_DIR, "youtube.json"), youtube_data)

        print("\n✅ Content generated and saved:")
        print("  - LinkedIn Post    → content/generated_content/linkedin_post.json")
        print("  - LinkedIn Article → content/generated_content/linkedin_article.json")
        print("  - X (Twitter)      → content/generated_content/twitter.json")
        print("  - YouTube          → content/generated_content/youtube.json")


    def generate_social_topics_with_influence(self, trend_influence, trends_data, industry, platform, trend_count, original_count):
        """Generate social media topics based on trend influence slider value"""
        print(f"🎯 DEBUG: Generating social topics with influence: {trend_influence}")
        print(f"📈 DEBUG: Trend topics: {trend_count}, Original topics: {original_count}")
        print(f"📱 DEBUG: Platform: {platform}")
        
        topics = []
        niche = self.load_json(NICHE_FILE)
        
        # Platform-specific context
        platform_contexts = {
            "linkedin-article": "professional, thought leadership, long-form analytical content",
            "linkedin-post": "professional, engaging, concise insights",
            "twitter": "concise, viral-worthy, trending, conversational",
            "youtube": "video-focused, tutorial-style, engaging storytelling"
        }
        platform_context = platform_contexts.get(platform, "engaging social media content")
        
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
        
        # Generate original topics (trend setters) - platform-specific
        for i in range(original_count):
            print(f"💡 DEBUG: Generating original topic {i+1}/{original_count} for {platform}")
            
            prompt = f"""
            You are a social media strategist creating ORIGINAL, trend-setting content for {platform}.
            
            Industry: {industry}
            Platform: {platform}
            Platform Style: {platform_context}
            Niche Context: {json.dumps(niche, indent=2) if niche else 'N/A'}
            
            {"Recent trends for context (create unique angles, don't copy):" if trends_data else ""}
            {json.dumps([t.get('title', '') for t in trends_data[:3]], indent=2) if trends_data else ''}
            
            Generate 1 ORIGINAL social media topic that:
            - Creates a NEW perspective or trend in {industry}
            - Is tailored for {platform} ({platform_context})
            - Goes beyond current discussions
            - Provides strategic, forward-thinking insights
            - Is engaging and shareable
            
            Platform-specific examples:
            - LinkedIn Article: "Why {industry} Needs to Rethink [Common Practice]: A Deep Dive"
            - LinkedIn Post: "The Hidden Cost of [Industry Standard] in {industry}"
            - X (Twitter): "Hot take: {industry} is missing [opportunity]. Here's why 🧵"
            - YouTube: "I Tested [New Approach] in {industry} for 30 Days - Results Shocked Me"
            
            Return ONLY in JSON:
            {{
              "topic": "Your original topic title tailored for {platform}",
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
                    "related_news": trends_data[:2] if trends_data else [],
                    "relevance_score": 85,
                    "type": "trend_setter"
                }
                topics.append(topic)
                print(f"✅ DEBUG: Added original topic: {topic['title']}")
                
            except Exception as e:
                print(f"❌ DEBUG: Error generating original topic: {e}")
                # Fallback original topics - platform-specific
                fallback_topics = {
                    "linkedin-article": [
                        f"Rethinking {industry}: A Strategic Imperative for 2025",
                        f"The Future of {industry}: Beyond Current Trends",
                        f"Why {industry} Leaders Are Missing the Bigger Picture"
                    ],
                    "linkedin-post": [
                        f"The {industry} shift nobody's talking about",
                        f"3 {industry} trends that will define 2025",
                        f"Why traditional {industry} approaches are failing"
                    ],
                    "twitter": [
                        f"Hot take: {industry} is broken. Here's how to fix it 🧵",
                        f"Everyone in {industry} is doing this wrong",
                        f"The {industry} playbook needs a rewrite"
                    ],
                    "youtube": [
                        f"I Tried the New {industry} Strategy - Results Were Shocking",
                        f"The {industry} Secret Nobody Tells You",
                        f"Why {industry} Experts Are Wrong About This"
                    ]
                }
                platform_fallbacks = fallback_topics.get(platform, fallback_topics["linkedin-post"])
                topic = {
                    "title": platform_fallbacks[i % len(platform_fallbacks)],
                    "related_news": trends_data[:2] if trends_data else [],
                    "relevance_score": 80,
                    "type": "trend_setter"
                }
                topics.append(topic)
                print(f"⚠️ DEBUG: Used fallback original topic: {topic['title']}")
        
        print(f"✅ DEBUG: Generated total {len(topics)} topics for {platform}")
        return topics


# Create alias for backward compatibility
PostGenerator = ContentPipeline

if __name__ == "__main__":
    pipeline = ContentPipeline()
    pipeline.run()

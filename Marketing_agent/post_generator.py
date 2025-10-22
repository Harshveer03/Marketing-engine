import os
import re
import json
from dotenv import load_dotenv
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings

load_dotenv()

NEWS_FILE = "./news/filtered_news.json"
TOPICS_FILE = "./topics/topics.json"
NICHE_FILE = "./niche/niche_icp.json"
OUTPUT_DIR = "./content/generated_content"
VECTOR_DB_DIR = "./vectordb"

def load_feedback_context():
    FEEDBACK_FILE = "./analytics/feedback_context.json"
    if not os.path.exists(FEEDBACK_FILE):
        print("⚠️ No feedback context found. Run feedback_loop.py first.")
        return {}
    with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


class ContentPipeline:
    def __init__(self, model="models/gemini-2.5-flash", embedding_model="nomic-embed-text"):
        self.llm = ChatGoogleGenerativeAI(model=model, temperature=0.7)
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        self.vectordb = FAISS.load_local(VECTOR_DB_DIR, self.embeddings, allow_dangerous_deserialization=True)
        self.feedback = load_feedback_context()

    # ---------- Utility ----------
    def load_json(self, path):
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

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

        print("\n🧠 Generated Topics:")
        for idx, t in enumerate(topics, start=1):
            print(f"{idx}. {t['title']} ({len(t['related_news'])} related articles)")

        return topics

    # ---------- Context Builders ----------
    def get_context(self, topic):
        niche = self.load_json(NICHE_FILE)
        query_text = topic["title"]
        pdf_docs = self.vectordb.similarity_search(query_text, k=10)
        pdf_context = "\n".join([doc.page_content for doc in pdf_docs])
        return niche, pdf_context

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
    def generate_linkedin(self, topic, related_news, niche, audience, tone, pdf_context):
        feedback_text = self.feedback.get("linkedin_feedback", "")
        pain_points = self._format_pain_points(niche)
        needs = self._format_needs(niche)
        prompt = f"""
        You are an AI assistant specialized in crafting high-impact LinkedIn posts for CXO and industry audiences.

        Your Task:
        Create a LinkedIn post on the topic: "{topic['title']}"

        Context Provided:
        -Industry: {niche.get("industry")}
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
            2. Ensure the content is engaging, authoritative, and strategically valuable for decision-makers.
            3. Highlight industry pain points, emerging needs, or opportunities with clarity.
            4. Incorporate storytelling or thought-leadership hooks to maximize engagement.
            5. Add 5–7 relevant, high-impact hashtags tailored to the industry and audience.
            6. Maintain a credible, CXO-level voice (avoid fluff, generic advice, or overselling).

        Goal:
        - The post should educate, provoke thought, and position the brand/author as a trusted authority in the space.

        Output in JSON:
        {{
          "linkedin": {{
            "caption": "...",
            "hashtags": ["#", "#"]
          }}
        }}
        """
        return self.clean_response(self.llm.invoke(prompt).content).get("linkedin", {})

    def generate_twitter(self, topic, related_news, niche, audience, tone, pdf_context):
        feedback_text = self.feedback.get("twitter_feedback", "")
        pain_points = ', '.join([p.get('challenge', '') for p in niche.get('customer_pain_points', [])])
        needs = self._format_needs(niche)
        prompt = f"""
        You are an AI assistant specialized in writing high-impact Twitter (X) posts for industry leaders.

        Task: Create a tweet on "{topic['title']}"
        Context:
        -Industry: {niche.get("industry")}
        -Pain Points: {pain_points}
        -Needs: {needs}
        -Audience: {audience}
        -Tone: {tone}

        Use the following performance feedback to guide brevity, tone, and structure:
        {feedback_text}

        Requirements:
            1. Must fit within 280 characters.
            2. Be punchy, concise, and attention-grabbing — avoid filler or generic phrasing.
            3. Deliver a sharp insight, challenge, or opportunity that resonates with CXO-level readers.
            4. Include 2–3 trending, relevant hashtags.
            5. Style should be thought-leadership driven (not just promotional).

        Goal:
        The tweet should spark conversation, showcase authority, and connect industry pain points with strategic opportunities in a way that encourages engagement.

        Output in JSON:
        {{
          "twitter": {{
            "tweet": "...",
            "hashtags": ["#", "#"]
          }}
        }}
        """
        return self.clean_response(self.llm.invoke(prompt).content).get("twitter", {})

    def generate_youtube(self, topic, related_news, niche, audience, tone, pdf_context):
        feedback_text = self.feedback.get("youtube_feedback", "")
        pain_points = self._format_pain_points(niche)
        needs = self._format_needs(niche)
        prompt = f"""
        You are an AI assistant specialized in creating YouTube video scripts and descriptions.

        Task: Generate a YouTube video intro and description for "{topic['title']}"

        Context:
        -Industry: {niche.get("industry")}
        -Pain Points: {pain_points}
        -Needs: {needs}
        -Audience: {audience}
        -Tone: {tone}

        Use the following performance feedback to guide video framing, SEO tone, and engagement style:
        {feedback_text}

        Requirements:
        1. Script Intro (30–45 seconds):
            - Hook the audience with a compelling, curiosity-driven opening line.
            - Briefly highlight industry pain points and why they matter now.
            - Introduce the value or solution your company/content will bring.
            - End with a reason to keep watching (tease what’s coming).
        2. Video Description (2–3 sentences):
            - Provide a clear, SEO-friendly summary of the video.
            - Emphasize value for the target audience and why they should watch.
            - Keep professional, concise, and engagement-driven.
        3. SEO Tags (5–7 keywords):
            - Must be relevant, search-optimized, and niche-specific.
            - Should cover industry trends, pain points, and opportunities.

        Goal:
            Produce an engaging, professional intro and description that not only retains viewers but also boosts discoverability on YouTube search.
        
        Output in JSON:
        {{
          "youtube": {{
            "script_intro": "...",
            "description": "...",
            "tags": ["tag1", "tag2"]
          }}
        }}
        """
        return self.clean_response(self.llm.invoke(prompt).content).get("youtube", {})

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

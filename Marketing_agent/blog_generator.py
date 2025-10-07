import os
import re
import json
from datetime import datetime
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from difflib import SequenceMatcher
import requests

load_dotenv()

VECTOR_DB_DIR = "./vectordb"
NICHE_FILE = "./niche/niche_icp.json"
USED_TOPICS_FILE = "./topics/used_blog_topics.json"
OUTPUT_DIR = "./content/blogs"
SERPAPI_KEY = os.getenv("SERPAPI_KEY")


class BlogGenerator:
    def __init__(self, model="models/gemini-2.5-flash", embedding_model="nomic-embed-text"):
        self.llm = ChatGoogleGenerativeAI(model=model, temperature=0.7)
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        self.vectordb = FAISS.load_local(VECTOR_DB_DIR, self.embeddings, allow_dangerous_deserialization=True)

    # ---------- Utility Loaders ----------
    def load_json(self, path):
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_used_topics(self):
        if os.path.exists(USED_TOPICS_FILE):
            with open(USED_TOPICS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def save_used_topics(self, topics):
        os.makedirs(os.path.dirname(USED_TOPICS_FILE), exist_ok=True)
        with open(USED_TOPICS_FILE, "w", encoding="utf-8") as f:
            json.dump(topics, f, indent=4, ensure_ascii=False)

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
        prompt = f"""
        You are a B2B SaaS marketing strategist and thought leadership architect.
        Your role is to craft long-form, insight-rich blog topics that challenge conventional thinking and offer a strategic lens on emerging shifts within the given niche.

        Input Context:
        ICP/Niche JSON:{json.dumps(niche, indent=2)}

        Your Task:
        Generate 1 unique, high-context, long-form blog topic that explores industry transformation, GTM evolution, or strategic inflection points.
        This topic should reflect a deep understanding of pain points, buyer psychology, market dynamics, and systemic inefficiencies revealed in the ICP/Niche data.

        Guidelines:

            1. The topic must sound like it belongs in a McKinsey, a16z, or Harvard Business Review-style publication — authoritative, specific, and original.

            2. Avoid generic titles like “Top Trends” or “The Future of X.” Focus on causality, systemic change, or strategic realignment.

            3. Use actionable phrasing that signals insight (e.g., "Rewiring," "Operationalizing," "Deconstructing," "Reframing," "Why GTM Models Fail", etc.).

            4. Keep the title concise (max 12–15 words) yet intellectually compelling — it should promise depth, not clickbait.

            5. Ensure the topic reflects real pain points or shifts from the ICP (e.g., scaling constraints, execution gaps, data fragmentation, buyer misalignment).
                
        Return ONLY in JSON:
        {{
          "topic": "Generated topic title"
        }}
        """

        response = self.llm.invoke(prompt).content
        match = re.search(r"\{.*\}", response, re.S)
        data = json.loads(match.group()) if match else {"topic": response.strip()}
        return data["topic"]

    # ---------- Similarity Context ----------
    def build_pdf_context(self, query_text):
        docs = self.vectordb.similarity_search(query_text, k=8)
        return "\n".join([doc.page_content for doc in docs])

    # ---------- Blog Generator ----------
    def generate_blog(self, topic, news_items, niche, pdf_context):
        prompt = f"""
        You are a B2B SaaS content strategist and industry analyst who writes deep, insight-rich long-form blogs designed for decision-makers, operators, and investors.
        Your job is to decode industry transformation — revealing why inefficiencies exist, how systems break down, and what strategic shifts define the next wave of growth.

        Task:
        Write a comprehensive, long-form analytical blog on the topic:
        "{topic}"

        Context Inputs:

            - Industry: {niche.get("industry")}

            - Key Pain Points: {[p['challenge'] for p in niche.get('customer_pain_points', [])]}

            - Customer Needs: {[n['need'] for n in niche.get('customer_needs', [])]}

            - Relevant News Articles: {json.dumps(news_items, indent=2)}

            - Reference Material (from ICP/Niche PDF): {pdf_context[:2000]}

        Writing Objectives

            1. Deliver a strategic narrative, not surface commentary — your writing should expose the root causes, hidden inefficiencies, and structural challenges in the market.

            2. Bridge macro trends (market shifts, capital cycles, AI adoption, GTM evolution) with micro realities (founder behavior, execution gaps, data fragmentation, operational misalignment).

            3. Blend data-backed reasoning and pattern recognition with storytelling that positions the reader as a strategic thinker.

            4. Every major claim or insight should implicitly answer:

                - Why is this happening now?

                - What are the systemic forces behind it?

                - What shift must companies make to adapt or win?

        Blog Requirements

            1. Length: 700–1000 words, structured and cohesive.

            2. Format:

                - Introduction: Contextualize the challenge and why it matters now.

                - Core Analysis: Break down root causes, system dynamics, and hidden frictions.

                - Strategic Solutions/Insights: Present clear frameworks, pivots, or models for GTM or operational advantage.

                - Conclusion: Forward-looking synthesis — what this shift means for SaaS leaders.

            3. Tone: Analytical, confident, and forward-thinking (McKinsey x a16z x Thought Leadership blend).

            4. Style:

                - Use cause-effect clarity (“because,” “driven by,” “due to”) to strengthen reasoning.

                - Avoid generic statements or motivational fluff.

                - No bullet lists or markdown formatting.

                - Avoid overt sales or brand promotion — focus on strategic substance.

            5. Integrate subtle nods to recent industry developments or evolving GTM playbooks where relevant.
        

        Output in JSON:
        {{
          "title": "{topic}",
          "outline": ["Intro", "Main Insight 1", "Main Insight 2", "Conclusion"],
          "blog": "Full text here..."
        }}
        """
        response = self.llm.invoke(prompt).content
        match = re.search(r"\{.*\}", response, re.S)
        data = json.loads(match.group()) if match else {"blog": response.strip()}
        return data

    # ---------- Execution Flow ----------
    def run(self, mode="manual"):
        niche = self.load_json(NICHE_FILE)
        used_topics = self.load_used_topics()

        if mode == "manual":
            topic = input("Enter your blog topic: ").strip()
            news_items = self.fetch_news(topic)
        else:
            print("🤖 Generating new blog topic from niche...")
            topic = self.generate_topic(niche)

            # avoid duplicate topics
            while any(self.is_similar(topic, t["title"]) for t in used_topics):
                print("⚠️ Duplicate topic detected, regenerating...")
                topic = self.generate_topic(niche)

            news_items = self.fetch_news(topic)
            used_topics.append({"title": topic, "generated_on": datetime.utcnow().isoformat()})
            self.save_used_topics(used_topics)

        pdf_context = self.build_pdf_context(topic)
        blog_data = self.generate_blog(topic, news_items, niche, pdf_context)

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(OUTPUT_DIR, f"blog_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(blog_data, f, indent=4, ensure_ascii=False)

        print(f"\n✅ Blog saved to: {output_path}")
        print(f"📝 Title: {blog_data.get('title', topic)}")
        return blog_data


if __name__ == "__main__":
    print("\n--- Blog Generator ---")
    mode = input("Choose mode (manual / automatic): ").strip().lower()
    if mode not in ["manual", "automatic"]:
        mode = "manual"
    gen = BlogGenerator()
    gen.run(mode)

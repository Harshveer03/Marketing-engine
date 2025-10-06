import os
import re
import json
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings

load_dotenv()

TOPICS_FILE = "./topics/topics.json"
NEWS_FILE = "./news/filtered_news.json"
NICHE_FILE = "./niche/niche_icp.json"
OUTPUT_DIR = "./content/generated_content"
VECTOR_DB_DIR = "./vectordb"


class ContentGenerator:
    def __init__(self, model="models/gemini-2.5-flash", embedding_model="nomic-embed-text"):
        self.llm = ChatGoogleGenerativeAI(model=model, temperature=0.7)
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        self.vectordb = FAISS.load_local(VECTOR_DB_DIR, self.embeddings, allow_dangerous_deserialization=True)

    def load_json(self, path):
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_context(self, topic_id: int):
        niche = self.load_json(NICHE_FILE)
        topics = self.load_json(TOPICS_FILE)
        news = self.load_json(NEWS_FILE)

        selected_topic = next((t for t in topics if t["id"] == topic_id), None)
        if not selected_topic:
            raise ValueError(f"Topic with id {topic_id} not found in {TOPICS_FILE}")

        related_news = [
            n for n in news
            if any(word.lower() in ((n.get("title") or "") + " " + (n.get("description") or "")).lower()
                   for word in selected_topic["title"].split())
        ]

        query_text = selected_topic["title"]
        pdf_docs = self.vectordb.similarity_search(query_text, k=10)
        pdf_context = "\n".join([doc.page_content for doc in pdf_docs])

        return niche, selected_topic, related_news, pdf_context

    def clean_response(self, response):
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", response, re.S)
            return json.loads(match.group()) if match else {}

    def _format_pain_points(self, niche):
        # Extract nested pain point details
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

    def generate_linkedin_content(self, topic, related_news, niche, audience, tone, pdf_context):
        pain_points_str = self._format_pain_points(niche)
        needs_str = self._format_needs(niche)

        prompt = f"""
        You are an AI assistant specialized in crafting high-impact LinkedIn posts for CXO and industry audiences.

        Your Task:
        Create a LinkedIn post on the topic: "{topic['title']}"

        Context Provided:

            -Industry: {niche.get("industry")}

            -Pain Points (with causes, explanations, indicators):
            {pain_points_str}

            -Needs: {needs_str}

            -Target Audience: {audience}

            -Desired Tone: {tone}

            -Related News: {json.dumps(related_news, indent=2, ensure_ascii=False)}

            -Reference Material (ICP/Niche PDF): {pdf_context}

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
            "hashtags": ["#", "#", "#"]
          }}
        }}
        """
        response = self.llm.invoke(prompt).content
        return self.clean_response(response).get("linkedin", {})

    def generate_twitter_content(self, topic, related_news, niche, audience, tone, pdf_context):
        # For Twitter, include only challenges to fit character limit
        pain_points_str = ', '.join([item.get('challenge', '') for item in niche.get('customer_pain_points', [])])
        needs_str = self._format_needs(niche)

        prompt = f"""
        You are an AI assistant specialized in writing high-impact Twitter (X) posts for industry leaders and professionals.

        Task:
        Create a tweet on the topic: "{topic['title']}"

        Context:

            -Industry: {niche.get("industry")}

            -Pain Points: {pain_points_str}

            -Needs: {needs_str}

            -Target Audience: {audience}

            -Desired Tone: {tone}

            -Related News: {json.dumps(related_news, indent=2, ensure_ascii=False)}

            -Reference Material (ICP/Niche PDF): {pdf_context}

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
        response = self.llm.invoke(prompt).content
        return self.clean_response(response).get("twitter", {})

    def generate_youtube_content(self, topic, related_news, niche, audience, tone, pdf_context):
        pain_points_str = self._format_pain_points(niche)
        needs_str = self._format_needs(niche)

        prompt = f"""
        You are an AI assistant specialized in creating YouTube video scripts and descriptions that position the brand as a thought leader.

        Task:
        Generate a YouTube video script intro and description for the topic: "{topic['title']}"

        Context:

            - Industry: {niche.get("industry")}

            - Pain Points (with causes, explanations, indicators): {pain_points_str}

            - Needs: {needs_str}

            - Target Audience: {audience}

            - Desired Tone: {tone}

            - Related News: {json.dumps(related_news, indent=2, ensure_ascii=False)}

            - Reference Material (ICP/Niche PDF): {pdf_context}

        Requirements:

        1. Script Intro (30–45 seconds):

            - Hook the audience with a compelling, curiosity-driven opening line.

            -Briefly highlight industry pain points and why they matter now.

            -Introduce the value or solution your company/content will bring.

            -End with a reason to keep watching (tease what’s coming).

        2. Video Description (2–3 sentences):

            - Provide a clear, SEO-friendly summary of the video.

            -Emphasize value for the target audience and why they should watch.

            -Keep professional, concise, and engagement-driven.

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
            "tags": ["tag1", "tag2", "tag3"]
          }}
        }}
        """
        response = self.llm.invoke(prompt).content
        return self.clean_response(response).get("youtube", {})

    def run(self, topic_id: int, audience: str, tone: str):
        niche, topic, related_news, pdf_context = self.get_context(topic_id)

        linkedin_post = self.generate_linkedin_content(topic, related_news, niche, audience, tone, pdf_context)
        twitter_post = self.generate_twitter_content(topic, related_news, niche, audience, tone, pdf_context)
        youtube_post = self.generate_youtube_content(topic, related_news, niche, audience, tone, pdf_context)

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # ✅ Simplified output structure
        linkedin_data = {
            "id": topic["id"],
            "title": topic["title"],
            "caption": linkedin_post.get("caption", ""),
            "hashtags": linkedin_post.get("hashtags", [])
        }

        twitter_data = {
            "id": topic["id"],
            "title": topic["title"],
            "caption": twitter_post.get("tweet", ""),
            "hashtags": twitter_post.get("hashtags", [])
        }

        youtube_data = {
            "id": topic["id"],
            "title": topic["title"],
            "script_intro": youtube_post.get("script_intro", ""),
            "caption": youtube_post.get("description", ""),
            "hashtags": youtube_post.get("tags", [])
        }

        linkedin_file = os.path.join(OUTPUT_DIR, "linkedin.json")
        twitter_file = os.path.join(OUTPUT_DIR, "twitter.json")
        youtube_file = os.path.join(OUTPUT_DIR, "youtube.json")

        with open(linkedin_file, "w", encoding="utf-8") as f:
            json.dump(linkedin_data, f, indent=4, ensure_ascii=False)

        with open(twitter_file, "w", encoding="utf-8") as f:
            json.dump(twitter_data, f, indent=4, ensure_ascii=False)

        with open(youtube_file, "w", encoding="utf-8") as f:
            json.dump(youtube_data, f, indent=4, ensure_ascii=False)

        print("✅ Simplified content generated and saved:")
        print("  -", linkedin_file)
        print("  -", twitter_file)
        print("  -", youtube_file)

        return linkedin_data, twitter_data, youtube_data



def main():
    generator = ContentGenerator()

    topics = generator.load_json(TOPICS_FILE)
    if not topics:
        print("⚠️ No topics found. Run topic_generator first.")
        return

    print("\nAvailable Topics:")
    for t in topics:
        print(f"- ID {t['id']}: {t['title']}")

    topic_id = int(input("\nEnter the Topic ID: ").strip())
    tone = input("Specify the tone (e.g., professional, casual, bold): ").strip()
    audience = input("Preferred target audience (e.g., CXOs, Founders, Marketers): ").strip()

    generator.run(topic_id, audience, tone)


if __name__ == "__main__":
    main()

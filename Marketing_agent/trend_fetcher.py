import os
import re
import json
import requests
from datetime import datetime
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

VECTOR_DB_DIR = "./vectordb"
OUTPUT_FILE = "./news/filtered_news.json"
SERPAPI_KEY = os.getenv("SERPAPI_KEY")  # must be in .env


class TrendFetcher:
    def __init__(self, model="models/gemini-2.0-flash", embedding_model="nomic-embed-text"):
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        self.vectordb = FAISS.load_local(
            VECTOR_DB_DIR, self.embeddings, allow_dangerous_deserialization=True
        )
        self.llm = ChatGoogleGenerativeAI(model=model, temperature=0)

    def build_queries(self, icp_json: dict) -> list:
        """Generate exactly 1 concise query using LLM"""
        industry = icp_json.get("industry", "")

        # Extract just 'challenge' text if pain points are dicts
        pain_points_data = icp_json.get("customer_pain_points", [])
        pain_points = ", ".join(
            [p["challenge"] if isinstance(p, dict) else str(p) for p in pain_points_data]
        )

        # Extract just 'need' text if customer_needs are dicts
        needs_data = icp_json.get("customer_needs", [])
        needs = ", ".join(
            [n["need"] if isinstance(n, dict) else str(n) for n in needs_data]
        )

        prompt = f"""
        Generate exactly 1 concise search query (2–3 words) related to {industry} trends, strategy, or customer challenges.
        Requirements:
        - The query MUST include either {industry}, {pain_points}, or {needs}.
        - Do not copy phrases directly from the input; rephrase into natural search terms.
        - Keep the query short (2–3 words max), distinct, and meaningful.
        - Avoid filler words, commas, or generic terms like "insights", "overview", "update".
        - Query should reflect a specific angle (e.g., a core challenge, need, or trend), not a vague phrase.

        Example:
        ["B2B SaaS GTM trends", "SaaS churn issues", "AI in SaaS", "SaaS CXO strategy", "SaaS growth 2025"]
        """

        response = self.llm.invoke(prompt).content.strip()
        match = re.search(r"\[.*\]", response, re.S)
        queries = json.loads(match.group()) if match else [f"{industry} trends"]

        cleaned = []
        for q in queries:
            q = re.sub(r"[^a-zA-Z0-9\s]", "", q).strip()
            words = q.split()
            if 2 <= len(words) <= 4:
                cleaned.append(" ".join(words))

        return cleaned if cleaned else [f"{industry} trends"]


    def fetch_serpapi(self, query: str, source: str = "google_news", num: int = 5) -> dict:
        """Fetch raw results from SerpAPI"""
        url = "https://serpapi.com/search"
        params = {"q": query, "engine": source, "num": num, "api_key": SERPAPI_KEY}
        resp = requests.get(url, params=params, timeout=20)
        return resp.json() if resp.status_code == 200 else {}

    def parse_results(self, data: dict, source: str) -> list:
        """Normalize SerpAPI results into a common schema"""
        results = []
        if not isinstance(data, dict):
            return results

        if source == "google_news":
            for art in data.get("news_results", []):
                results.append({
                    "title": art.get("title"),
                    "url": art.get("link"),
                    "description": art.get("snippet"),
                    "publishedAt": art.get("date"),
                    "source": "Google News"
                })

        elif source == "reddit":
            for post in data.get("organic_results", []):
                results.append({
                    "title": post.get("title"),
                    "url": post.get("link"),
                    "description": post.get("snippet", ""),
                    "publishedAt": datetime.utcnow().isoformat(),
                    "source": "Reddit"
                })

        elif source == "twitter":
            for tweet in data.get("organic_results", []):
                results.append({
                    "title": tweet.get("title"),
                    "url": tweet.get("link"),
                    "description": tweet.get("snippet", ""),
                    "publishedAt": datetime.utcnow().isoformat(),
                    "source": "Twitter"
                })

        elif source == "linkedin":
            for post in data.get("organic_results", []):
                results.append({
                    "title": post.get("title"),
                    "url": post.get("link"),
                    "description": post.get("snippet", ""),
                    "publishedAt": datetime.utcnow().isoformat(),
                    "source": "LinkedIn"
                })

        elif source == "youtube":
            for vid in data.get("video_results", []):
                results.append({
                    "title": vid.get("title"),
                    "url": vid.get("link"),
                    "description": vid.get("snippet"),
                    "publishedAt": vid.get("date", datetime.utcnow().isoformat()),
                    "source": "YouTube"
                })

        return results

    def relevance_filter(self, items: list, icp_json: dict, top_k: int = 10, threshold: float = 0.65) -> list:
        """Filter results using ICP keywords + vector similarity with threshold"""
        keywords = [icp_json.get("industry", "").lower()]

        # Extract challenge text if available
        pain_points_data = icp_json.get("customer_pain_points", [])
        for p in pain_points_data:
            if isinstance(p, dict):
                challenge = p.get("challenge", "")
                if challenge:
                    keywords.append(challenge.lower())
            elif isinstance(p, str):
                keywords.append(p.lower())

        # Extract need text if available
        needs_data = icp_json.get("customer_needs", [])
        for n in needs_data:
            if isinstance(n, dict):
                need = n.get("need", "")
                if need:
                    keywords.append(need.lower())
            elif isinstance(n, str):
                keywords.append(n.lower())

        filtered = []
        for item in items:
            text = f"{item.get('title','')} {item.get('description','')}".lower()

            keyword_hit = any(kw in text for kw in keywords if kw)
            docs_and_scores = self.vectordb.similarity_search_with_score(text, k=1)
            semantic_hit = docs_and_scores and docs_and_scores[0][1] >= threshold

            if keyword_hit or semantic_hit:
                filtered.append({
                    **item,
                    "relevance_context": docs_and_scores[0][0].page_content if docs_and_scores else "",
                    "similarity_score": float(docs_and_scores[0][1]) if docs_and_scores else None,
                })

        return filtered[:top_k]


    def run(self, icp_json_path="./niche/niche_icp.json"):
        with open(icp_json_path, "r") as f:
            icp_json = json.load(f)

        queries = self.build_queries(icp_json)
        print(f"🔍 Queries: {queries}")

        all_items = []
        for q in queries:
            raw = self.fetch_serpapi(q, source="google_news")
            print(f"📥 google_news returned {len(raw) if isinstance(raw, dict) else 0} keys for query '{q}'")
            items = self.parse_results(raw, source="google_news")
            print(f"   Parsed {len(items)} items from google_news")
            all_items.extend(items)

        # ✅ only top 10 will be returned
        filtered = self.relevance_filter(all_items, icp_json, top_k=10)

        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(filtered, f, indent=4)

        print(f"✅ Saved {len(filtered)} relevant trends to {OUTPUT_FILE}")
        return filtered


if __name__ == "__main__":
    fetcher = TrendFetcher()
    results = fetcher.run()
    for r in results:
        print(f"- [{r['source']}] {r['title']} ({r['url']})")

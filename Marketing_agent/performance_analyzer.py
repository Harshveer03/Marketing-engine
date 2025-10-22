import os
import json
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

PERFORMANCE_FILE = "./analytics/performance_data.json"
INSIGHTS_FILE = "./analytics/performance_insights.json"


class LLMPerformanceAnalyzer:
    def __init__(self, model="models/gemini-2.5-flash"):
        self.llm = ChatGoogleGenerativeAI(model=model, temperature=0.3)

    def load_json(self, path):
        if not os.path.exists(path):
            print(f"⚠️ No file found at {path}")
            return []
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def preprocess_data(self, data):
        """Normalize performance data — ensure each record has required keys."""
        cleaned = []
        for record in data:
            cleaned.append({
                "platform": record.get("platform", "unknown"),
                "title": record.get("title", "Untitled"),
                "metrics": record.get("metrics", {}),
                "timestamp": record.get("timestamp", "")
            })
        return cleaned

    def analyze_with_llm(self, data):
        prompt = f"""
        You are an AI performance analyst for marketing content.

        Analyze the following dataset of **LinkedIn, Twitter, YouTube, and Blog** content performance:
        Each record includes title, platform, and engagement metrics (impressions, likes, comments, shares, engagement_rate).

        Dataset:
        {json.dumps(data, indent=2, ensure_ascii=False)}

        Tasks:
        1. Detect key patterns: what types of **titles**, **tones**, or **topics** drive higher engagement.
        2. Identify which **platforms perform best**, and what kind of content succeeds there.
        3. List the **top 3 best-performing titles** overall and briefly explain their success factors.
        4. Suggest actionable, data-driven **recommendations** for each platform.

        Format output strictly as valid JSON:
        {{
          "summary": {{
            "platforms": {{
              "linkedin": {{
                "avg_engagement": 0.0,
                "top_titles": ["...", "..."],
                "insights": "...",
                "recommendations": "..."
              }},
              "twitter": {{
                "avg_engagement": 0.0,
                "top_titles": ["...", "..."],
                "insights": "...",
                "recommendations": "..."
              }},
              "youtube": {{
                "avg_engagement": 0.0,
                "top_titles": ["...", "..."],
                "insights": "...",
                "recommendations": "..."
              }},
              "blog": {{
                "avg_engagement": 0.0,
                "top_titles": ["...", "..."],
                "insights": "...",
                "recommendations": "..."
              }}
            }},
            "global_insights": {{
              "top_performing_titles": ["...", "...", "..."],
              "common_success_factors": "...",
              "overall_recommendation": "..."
            }}
          }}
        }}
        """

        response = self.llm.invoke(prompt).content

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            print("⚠️ LLM returned invalid JSON, attempting to repair...")
            start = response.find("{")
            end = response.rfind("}") + 1
            try:
                return json.loads(response[start:end])
            except Exception:
                print("❌ Could not parse LLM response properly.")
                return {"summary": {"platforms": {}, "global_insights": {}}}

    def run(self):
        print("📈 Running AI-driven performance analysis...\n")

        data = self.load_json(PERFORMANCE_FILE)
        if not data:
            print("⚠️ No performance data found.")
            return

        processed_data = self.preprocess_data(data)
        analysis = self.analyze_with_llm(processed_data)

        os.makedirs(os.path.dirname(INSIGHTS_FILE), exist_ok=True)
        with open(INSIGHTS_FILE, "w", encoding="utf-8") as f:
            json.dump(analysis, f, indent=4, ensure_ascii=False)

        print(f"✅ AI-generated performance insights saved to {INSIGHTS_FILE}\n")

        # 🧠 Print summary for quick visibility
        summary = analysis.get("summary", {})
        platforms = summary.get("platforms", {})
        for platform, info in platforms.items():
            print(f"📊 {platform.capitalize()}")
            print(f"   Avg Engagement: {info.get('avg_engagement', 0.0)}")
            print(f"   Insights: {info.get('insights', 'N/A')}")
            print(f"   Recommendations: {info.get('recommendations', 'N/A')}")
            top_titles = info.get('top_titles', [])
            if top_titles:
                print(f"   Top Titles: {', '.join(top_titles)}")
            print()

        global_summary = summary.get("global_insights", {})
        print("🌍 Global Insights:")
        print(f"   🔝 Top Titles: {', '.join(global_summary.get('top_performing_titles', []))}")
        print(f"   💡 Common Success Factors: {global_summary.get('common_success_factors', 'N/A')}")
        print(f"   🧭 Overall Recommendation: {global_summary.get('overall_recommendation', 'N/A')}")


if __name__ == "__main__":
    analyzer = LLMPerformanceAnalyzer()
    analyzer.run()

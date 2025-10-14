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

    def analyze_with_llm(self, data):
        prompt = f"""
        You are an expert AI performance analyst for marketing content.

        Analyze the following dataset of post and blog performances across LinkedIn, Twitter, YouTube, and Blog:
        Each record includes title, metrics, platform, and timestamp.

        Dataset:
        {json.dumps(data, indent=2, ensure_ascii=False)}

        Your tasks:
        1. Identify performance patterns — what types of **titles**, **tones**, or **topics** yield high engagement.
        2. Determine which **platforms perform best**, and what kind of content thrives there.
        3. Find the **top 3 highest-performing titles overall** and explain why they worked.
        4. Suggest **data-driven recommendations** for each platform to improve future content.

        Format your response strictly as valid JSON:
        {{
          "summary": {{
            "platforms": {{
              "linkedin": {{
                "avg_engagement": 0.0,
                "insights": "...",
                "recommendations": "...",
                "top_titles": ["...", "..."]
              }},
              "twitter": {{
                "avg_engagement": 0.0,
                "insights": "...",
                "recommendations": "...",
                "top_titles": ["...", "..."]
              }},
              "youtube": {{
                "avg_engagement": 0.0,
                "insights": "...",
                "recommendations": "...",
                "top_titles": ["...", "..."]
              }},
              "blog": {{
                "avg_engagement": 0.0,
                "insights": "...",
                "recommendations": "...",
                "top_titles": ["...", "..."]
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
            print("⚠️ LLM returned invalid JSON, attempting to clean...")
            start = response.find("{")
            end = response.rfind("}") + 1
            return json.loads(response[start:end])

    def run(self):
        print("📈 Running LLM-driven performance analysis...")

        data = self.load_json(PERFORMANCE_FILE)
        if not data:
            print("⚠️ No performance data found.")
            return

        analysis = self.analyze_with_llm(data)

        os.makedirs(os.path.dirname(INSIGHTS_FILE), exist_ok=True)
        with open(INSIGHTS_FILE, "w", encoding="utf-8") as f:
            json.dump(analysis, f, indent=4, ensure_ascii=False)

        print(f"✅ AI-generated performance insights saved to {INSIGHTS_FILE}\n")

        # Display summary in readable form
        for platform, info in analysis["summary"]["platforms"].items():
            print(f"📊 {platform.capitalize()}")
            print(f"   Avg Engagement: {info.get('avg_engagement')}")
            print(f"   Insights: {info.get('insights')}")
            print(f"   Recommendations: {info.get('recommendations')}")
            print(f"   Top Titles: {', '.join(info.get('top_titles', []))}\n")

        global_summary = analysis["summary"]["global_insights"]
        print("🌍 Global Insights:")
        print(f"   🔝 Top Titles: {', '.join(global_summary['top_performing_titles'])}")
        print(f"   💡 Common Success Factors: {global_summary['common_success_factors']}")
        print(f"   🧭 Overall Recommendation: {global_summary['overall_recommendation']}")


if __name__ == "__main__":
    analyzer = LLMPerformanceAnalyzer()
    analyzer.run()

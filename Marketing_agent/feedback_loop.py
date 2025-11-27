import os
import json
from dotenv import load_dotenv

INSIGHTS_FILE = "./generated/analytics/performance_insights.json"
FEEDBACK_CONTEXT_FILE = "./generated/analytics/feedback_context.json"

class FeedbackLoop:
    def __init__(self):
        load_dotenv()

    def load_insights(self):
        if not os.path.exists(INSIGHTS_FILE):
            print("⚠️ No performance insights found. Run performance_analyzer first.")
            return {}
        with open(INSIGHTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    def build_feedback_context(self, insights):
        summary = insights.get("summary", {})
        platforms = summary.get("platforms", {})
        global_info = summary.get("global_insights", {})

        context = {
            "linkedin_feedback": platforms.get("linkedin", {}).get("recommendations", ""),
            "twitter_feedback": platforms.get("twitter", {}).get("recommendations", ""),
            "youtube_feedback": platforms.get("youtube", {}).get("recommendations", ""),
            "blog_feedback": platforms.get("blog", {}).get("recommendations", ""),
            "global_success_factors": global_info.get("common_success_factors", ""),
            "overall_recommendation": global_info.get("overall_recommendation", "")
        }

        return context

    def save_context(self, context):
        os.makedirs(os.path.dirname(FEEDBACK_CONTEXT_FILE), exist_ok=True)
        with open(FEEDBACK_CONTEXT_FILE, "w", encoding="utf-8") as f:
            json.dump(context, f, indent=4, ensure_ascii=False)
        print(f"✅ Feedback context saved to {FEEDBACK_CONTEXT_FILE}")

    def run(self):
        print("🔁 Building adaptive feedback context from performance insights...\n")

        insights = self.load_insights()
        if not insights:
            print("⚠️ No insights to process.")
            return

        context = self.build_feedback_context(insights)
        self.save_context(context)

        print("\n🧠 Feedback Context Summary:")
        for k, v in context.items():
            print(f"• {k}: {v[:120]}{'...' if len(v) > 120 else ''}")


if __name__ == "__main__":
    loop = FeedbackLoop()
    loop.run()

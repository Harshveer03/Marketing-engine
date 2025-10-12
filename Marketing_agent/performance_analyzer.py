import os
import json
from collections import defaultdict, Counter
from datetime import datetime

PERFORMANCE_FILE = "./analytics/performance_data.json"
INSIGHTS_FILE = "./analytics/adaptive_insights.json"

def load_performance_data(path):
    if not os.path.exists(path):
        print("⚠️ Performance data not found.")
        return []
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            print("⚠️ Could not parse performance data.")
            return []

def analyze_platform(records):
    if not records:
        return {}

    # Compute average engagement
    avg_engagement = sum(r["metrics"]["engagement_rate"] for r in records) / len(records)
    
    # Identify top-performing titles / topics
    sorted_by_engagement = sorted(records, key=lambda r: r["metrics"]["engagement_rate"], reverse=True)
    top_titles = [r["title"] for r in sorted_by_engagement[:3]]  # top 3
    avoid_titles = [r["title"] for r in sorted_by_engagement[-3:]]  # bottom 3
    
    # Optional: analyze hashtags if available
    hashtags = Counter()
    for r in records:
        if "hashtags" in r:
            hashtags.update(r["hashtags"])
    top_hashtags = [tag for tag, _ in hashtags.most_common(5)]
    low_hashtags = [tag for tag, _ in hashtags.most_common()[-5:]]

    return {
        "avg_engagement": round(avg_engagement, 3),
        "top_titles": top_titles,
        "avoid_titles": avoid_titles,
        "top_hashtags": top_hashtags,
        "low_hashtags": low_hashtags
    }

def generate_adaptive_insights(data):
    platform_records = defaultdict(list)
    for record in data:
        platform_records[record["platform"]].append(record)

    insights = {}
    for platform, records in platform_records.items():
        insights[platform] = analyze_platform(records)
    return insights

def save_insights(insights, path=INSIGHTS_FILE):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(insights, f, indent=4, ensure_ascii=False)
    print(f"✅ Adaptive insights saved to {path}")

def main():
    print("📊 Loading historical performance data...")
    data = load_performance_data(PERFORMANCE_FILE)
    if not data:
        print("⚠️ No data to analyze. Exiting.")
        return

    print("🧠 Analyzing engagement metrics...")
    insights = generate_adaptive_insights(data)
    
    save_insights(insights)

    # Optional: print summary for quick inspection
    for platform, info in insights.items():
        print(f"\n📌 {platform.upper()} Insights:")
        print(f"Avg Engagement Rate: {info['avg_engagement']}")
        print(f"Prioritize Titles: {info['top_titles']}")
        print(f"Avoid Titles: {info['avoid_titles']}")
        print(f"Top Hashtags: {info['top_hashtags']}")
        print(f"Low Hashtags: {info['low_hashtags']}")

if __name__ == "__main__":
    main()

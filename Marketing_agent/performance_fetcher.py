import os
import json
import random
from datetime import datetime

PERFORMANCE_DIR = "./analytics"
CONTENT_DIR = "./content/generated_content"
BLOG_DIR = "./content/blogs"
OUTPUT_FILE = os.path.join(PERFORMANCE_DIR, "performance_data.json")

os.makedirs(PERFORMANCE_DIR, exist_ok=True)


def load_json_safe(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            print(f"⚠️ Could not parse {path}")
            return None


def extract_metadata(platform, data):
    """Extracts topic_id and title safely from JSON content."""
    if not data:
        return None, "Untitled"

    if platform in ["linkedin", "twitter", "youtube"]:
        return data.get("id"), data.get("title", "Untitled")

    if platform == "blog":
        # Handle blog JSON structure — no fixed filename or topic_id
        if isinstance(data, dict):
            return None, data.get("title", "Untitled")
        else:
            print("⚠️ Blog JSON not a valid object, skipping.")
            return None, "Untitled"

    return None, "Untitled"


def generate_fake_metrics():
    """Simulates realistic engagement metrics."""
    impressions = random.randint(2000, 20000)
    likes = random.randint(40, 300)
    comments = random.randint(5, 60)
    shares = random.randint(5, 50)
    engagement_rate = round((likes + comments + shares) / impressions, 3)
    return {
        "impressions": impressions,
        "likes": likes,
        "comments": comments,
        "shares": shares,
        "engagement_rate": engagement_rate
    }


def collect_metrics():
    platforms = ["linkedin", "twitter", "youtube", "blog"]
    all_data = []

    for platform in platforms:
        if platform == "blog":
            # Get latest blog file
            blog_files = [f for f in os.listdir(BLOG_DIR) if f.startswith("blog_") and f.endswith(".json")]
            if not blog_files:
                print("⚠️ No blog files found, skipping blog metrics.")
                continue

            latest_blog = max(blog_files, key=lambda f: os.path.getmtime(os.path.join(BLOG_DIR, f)))
            data = load_json_safe(os.path.join(BLOG_DIR, latest_blog))
        else:
            file_path = os.path.join(CONTENT_DIR, f"{platform}.json")
            data = load_json_safe(file_path)

        if not data:
            print(f"⚠️ Skipping {platform} — no content found.")
            continue

        topic_id, title = extract_metadata(platform, data)
        metrics = generate_fake_metrics()

        record = {
            "topic_id": topic_id,
            "platform": platform,
            "title": title,
            "metrics": metrics,
            "timestamp": datetime.utcnow().isoformat()
        }

        all_data.append(record)
        print(f"📊 Collected metrics for {platform}: {title}")

    if all_data:
        existing = load_json_safe(OUTPUT_FILE) or []
        existing.extend(all_data)

        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=4, ensure_ascii=False)

        print(f"✅ Saved {len(all_data)} new records to {OUTPUT_FILE}")
    else:
        print("⚠️ No metrics collected — check source files.")


if __name__ == "__main__":
    collect_metrics()

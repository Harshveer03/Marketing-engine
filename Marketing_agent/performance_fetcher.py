import os
import json
import random
from datetime import datetime

PERFORMANCE_DIR = "./generated/analytics"
CONTENT_DIR = "./generated/content/social"
BLOGS_FILE = "./generated/content/blogs/blogs.json"
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
    """Extracts title safely from platform JSON content."""
    if not data:
        return "Untitled"

    if platform in ["linkedin", "twitter", "youtube"]:
        if isinstance(data, list):
            # Take the last post entry if multiple exist
            return data[-1].get("title", "Untitled")
        elif isinstance(data, dict):
            return data.get("title", "Untitled")

    if platform == "blog":
        if isinstance(data, dict):
            return data.get("title", "Untitled")
        elif isinstance(data, list):
            # For blogs.json (list of all blogs)
            return [b.get("title", "Untitled") for b in data]

    return "Untitled"


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
            blog_data = load_json_safe(BLOGS_FILE)
            if not blog_data or not isinstance(blog_data, list):
                print("⚠️ No blogs found, skipping blog metrics.")
                continue

            for blog in blog_data:
                title = blog.get("title", "Untitled")
                metrics = generate_fake_metrics()

                record = {
                    "platform": platform,
                    "title": title,
                    "metrics": metrics,
                    "timestamp": datetime.utcnow().isoformat()
                }

                all_data.append(record)
                print(f"📊 Collected metrics for blog: {title}")

        else:
            file_path = os.path.join(CONTENT_DIR, f"{platform}.json")
            data = load_json_safe(file_path)
            if not data:
                print(f"⚠️ Skipping {platform} — no content found.")
                continue

            # Handle both list and dict data formats
            if isinstance(data, list):
                for entry in data:
                    title = entry.get("title", "Untitled")
                    metrics = generate_fake_metrics()
                    record = {
                        "platform": platform,
                        "title": title,
                        "metrics": metrics,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    all_data.append(record)
                    print(f"📊 Collected metrics for {platform}: {title}")
            else:
                title = data.get("title", "Untitled")
                metrics = generate_fake_metrics()
                record = {
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

# src/extract_issues.py

import csv
import json
import re
from collections import Counter
import os
from pathlib import Path


# ---------- Config ----------
def get_project_root():
    return Path(__file__).resolve().parents[1]


CSV_PATH = get_project_root() / "data" / "train.csv"
OUT_JSON = get_project_root() / "data" / "suggestions_from_test.json"
SAMPLE_ROWS = 20000
SAMPLE_PER_CATEGORY = 200
MIN_PHRASE_COUNT = 2
# ----------------------------

categories = {
    "battery": ["battery", "charge", "charging", "power", "drain", "drains", "battery life", "battery backup"],
    "camera": ["camera", "photo", "image", "picture", "selfie", "low light", "blurry", "focus", "autofocus"],
    "delivery": ["delivery", "delivered", "shipping", "delay", "late", "courier", "shipment", "shipped"],
    "performance": ["slow", "lag", "freeze", "hang", "performance", "speed", "slowly", "sluggish"],
    "heating": ["heat", "hot", "heating", "overheat", "warm"],
    "audio": ["sound", "audio", "speaker", "volume", "distortion", "mic", "microphone", "bass", "treble"],
    "screen": ["screen", "display", "brightness", "touch", "pixel", "flicker", "resolution"],
    "build_quality": ["broken", "damage", "defect", "scratch", "crack", "fragile", "durable", "sturdy"],
    "packaging": ["package", "packaging", "box", "damaged packaging", "wrapped", "wrap", "bubble"],
    "price": ["price", "expensive", "costly", "overpriced", "value", "worth"],
    "connectivity": ["wifi", "wi-fi", "bluetooth", "connect", "disconnect", "signal", "network", "hotspot"],
    "software": ["software", "app", "ui", "interface", "bug", "glitch", "update", "crash", "patch"]
}

# Normalize keywords
for k in categories:
    categories[k] = [kw.lower() for kw in categories[k]]


def norm_text(text):
    """Normalize text for analysis"""
    text = (text or "").lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [t for t in text.split() if t]
    return tokens


def extract_local_ngrams(tokens, idx, sizes=(2, 3, 4)):
    """Extract n-grams around a keyword"""
    ngs = []
    L = len(tokens)
    for size in sizes:
        for start in range(max(0, idx - size + 1), idx + 1):
            end = start + size
            if end <= L:
                ng = " ".join(tokens[start:end])
                ngs.append(ng)
    return ngs


def extract_issues():
    """Extract issues and generate suggestions"""

    # Check if CSV exists
    if not CSV_PATH.exists():
        print(f"⚠️ Training data not found at: {CSV_PATH}")
        print("Creating default suggestions...")
        return create_default_suggestions()

    print(f"Processing {CSV_PATH}...")

    # Counters
    cat_counts = {c: 0 for c in categories}
    cat_phrases = {c: Counter() for c in categories}
    cat_samples = {c: [] for c in categories}

    # Read CSV
    try:
        with open(CSV_PATH, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i >= SAMPLE_ROWS:
                    break

                # Get review text
                review = ""
                if len(row) >= 3:
                    review = row[2]
                elif len(row) >= 2:
                    review = row[1]
                else:
                    continue

                tokens = norm_text(review)
                if not tokens:
                    continue

                joined = " ".join(tokens)

                # Check each category
                for cat, kws in categories.items():
                    matched = False
                    for kw in kws:
                        if kw in joined:
                            matched = True
                            break

                    if not matched:
                        continue

                    # Matched category
                    cat_counts[cat] += 1
                    if len(cat_samples[cat]) < SAMPLE_PER_CATEGORY:
                        cat_samples[cat].append(review)

                    # Extract n-grams
                    for idx, tok in enumerate(tokens):
                        for kw in categories[cat]:
                            if kw in tok:
                                for ng in extract_local_ngrams(tokens, idx, sizes=(2, 3, 4)):
                                    cat_phrases[cat][ng] += 1

    except Exception as e:
        print(f"⚠️ Error reading CSV: {e}")
        return create_default_suggestions()

    # Build suggestion templates
    suggestion_templates = {
        "battery": [
            "Improve battery life and endurance",
            "Optimize power consumption and background processes",
            "Add or improve fast-charging support"
        ],
        "camera": [
            "Enhance camera performance in low-light conditions",
            "Improve image stabilization and autofocus",
            "Upgrade image processing and reduce noise"
        ],
        "delivery": [
            "Improve delivery speed and logistics coordination",
            "Use better packaging to prevent transit damage",
            "Provide accurate tracking and timely status updates"
        ],
        "performance": [
            "Optimize system performance to reduce lag and freezing",
            "Improve memory and resource management for smoother multitasking",
            "Fix performance bottlenecks in common use-cases"
        ],
        "heating": [
            "Improve thermal management to reduce overheating",
            "Optimize CPU/GPU usage to lower device temperature",
            "Enhance heat dissipation design"
        ],
        "audio": [
            "Improve speaker clarity and reduce distortion",
            "Enhance microphone quality for clearer calls",
            "Tune audio output for better bass and treble balance"
        ],
        "screen": [
            "Improve display brightness and outdoor visibility",
            "Enhance touch responsiveness and reduce ghost touches",
            "Use higher-quality panels for better color accuracy"
        ],
        "build_quality": [
            "Use more durable materials to reduce cracking and scratching",
            "Improve assembly quality to avoid defects",
            "Increase durability testing during manufacturing"
        ],
        "packaging": [
            "Improve protective packaging materials and cushioning",
            "Ensure boxes are sealed and resistant to transit handling",
            "Label fragile items and use tamper-evident seals"
        ],
        "price": [
            "Reevaluate pricing strategy to offer better value",
            "Consider bundling accessories or offering discounts",
            "Provide clearer feature-to-price comparisons"
        ],
        "connectivity": [
            "Fix WiFi/Bluetooth stability issues and reduce disconnects",
            "Improve antenna design for stronger signal",
            "Optimize network handover and hotspot stability"
        ],
        "software": [
            "Fix recurring software bugs and crashes",
            "Improve UI responsiveness and remove bloatware",
            "Provide regular, tested updates and clear changelogs"
        ]
    }

    # Build final map
    final_map = {}
    for cat in categories:
        top_ph = [p for p, c in cat_phrases[cat].most_common(100) if c >= MIN_PHRASE_COUNT]
        if len(top_ph) == 0:
            top_ph = categories[cat][:10]

        final_map[cat] = {
            "count_sampled_matches": cat_counts[cat],
            "evidence_phrases": top_ph[:50],
            "suggestions": suggestion_templates.get(cat, ["General improvements recommended"]),
            "sample_reviews": cat_samples[cat][:10]
        }

    # Write JSON
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(final_map, f, indent=2, ensure_ascii=False)

    print(f"✓ Wrote: {OUT_JSON}")
    print(f"✓ Sample counts: {cat_counts}")
    return final_map


def create_default_suggestions():
    """Create default suggestions when no data available"""
    suggestion_templates = {
        "battery": ["Improve battery life", "Optimize power consumption", "Add fast-charging support"],
        "camera": ["Enhance camera quality", "Improve low-light performance", "Better image stabilization"],
        "delivery": ["Improve delivery speed", "Better packaging", "Accurate tracking"],
        "performance": ["Optimize performance", "Reduce lag", "Better resource management"],
        "heating": ["Improve thermal management", "Reduce overheating", "Better heat dissipation"],
        "audio": ["Improve audio quality", "Better speaker clarity", "Enhanced microphone"],
        "screen": ["Better display quality", "Improve brightness", "Enhanced touch response"],
        "build_quality": ["Use durable materials", "Improve build quality", "Better assembly"],
        "packaging": ["Better packaging", "Protective materials", "Secure sealing"],
        "price": ["Better value proposition", "Competitive pricing", "Clear pricing"],
        "connectivity": ["Stable WiFi/Bluetooth", "Better signal", "Improved connectivity"],
        "software": ["Fix software bugs", "Better UI", "Regular updates"]
    }

    final_map = {}
    for cat, suggestions in suggestion_templates.items():
        final_map[cat] = {
            "count_sampled_matches": 0,
            "evidence_phrases": categories.get(cat, []),
            "suggestions": suggestions,
            "sample_reviews": []
        }

    # Write JSON
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(final_map, f, indent=2, ensure_ascii=False)

    print(f"✓ Created default suggestions: {OUT_JSON}")
    return final_map


if __name__ == "__main__":
    print("Extracting issues and generating suggestions...")
    extract_issues()